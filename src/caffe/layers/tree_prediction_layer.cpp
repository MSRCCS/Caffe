#include <algorithm>
#include <vector>
#include <float.h>

#include "caffe/layers/tree_prediction_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TreePredictionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    Layer<Dtype>::LayerSetUp(bottom, top);
    tree_.read(this->layer_param().treeprediction_param().tree().c_str());
    if (tree_.groups() == 1)
        LOG(WARNING) << "With only a single group in the tree, it is more efficient to use ArgmaxLayer instead of TreePredictionLayer";
    threshold_ = this->layer_param().treeprediction_param().threshold();
    has_map_ = this->layer_param_.treeprediction_param().has_map();
    if (has_map_) {
        CHECK(!this->layer_param_.treeprediction_param().full_map()) << 
            "Must not specify both map and full_map";
        read_map(this->layer_param().treeprediction_param().map().c_str(), tree_.nodes(), label_map_);
    } else if (this->layer_param_.treeprediction_param().full_map()) {
        label_map_.Reshape({ tree_.nodes() });
        has_map_ = true;
        for (int i = 0; i < label_map_.count(); ++i)
            label_map_.mutable_cpu_data()[i] = i;
    }

#ifndef CPU_ONLY
    // Pre-fetch data
    if (Caffe::mode() == Caffe::GPU) {
        if (has_map_) {
            label_map_.mutable_gpu_data();
            tree_.parent_.mutable_gpu_data();
        } else {
            tree_.group_size_.mutable_gpu_data();
            tree_.group_offset_.mutable_gpu_data();
            tree_.child_.mutable_gpu_data();
        }
    }
#endif
}

template <typename Dtype>
void TreePredictionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());

  int channels = bottom[0]->shape(axis_);

  // This may requires a reshape layer to reshape to CxA before tree_prediction
  CHECK(channels == tree_.nodes()) << "Channel count: " << channels << " must match tree node count: " << tree_.nodes();
  outer_num_ = bottom[0]->count(0, axis_);
  inner_num_ = bottom[0]->count(axis_ + 1);

  auto shape = bottom[0]->shape();
  shape[axis_] = 1;
  top[0]->Reshape(shape); // argmax among hierarchical probabilities
  if (has_map_) {
      CHECK_EQ(axis_, 1)
          << "Axis must be 1 (other axes are not yet supported with map)";
      if (top.size() == 3)
          top[2]->Reshape(shape); // the max values associated with the argmax in top[0]
      shape[axis_] = label_map_.count();
  }
  if (top.size() >= 2)
      top[1]->Reshape(shape);
  prob_.Reshape(shape); // hierarchical class probability
}

template <typename Dtype>
void TreePredictionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  auto argmax_data = top[0]->mutable_cpu_data();
  auto top_data = prob_.mutable_cpu_data();
  auto prob_data = bottom[0]->cpu_data();
  int channels = bottom[0]->shape(axis_);

  if (has_map_) {
      Dtype* max_data = NULL;
      if (top.size() == 3)
          max_data = top[2]->mutable_cpu_data();

      auto parent_data = tree_.parent_.cpu_data();
      auto label_count = label_map_.count();
      auto label_data = label_map_.cpu_data();

      // Find hierarchical probabilities for labels in the map
#pragma omp parallel for
      for (int index = 0; index < outer_num_ * label_count * inner_num_; ++index) {
          int s = index % inner_num_;
          int i = (index / inner_num_) % label_count;
          int n = (index / inner_num_) / label_count;

          int label_value = label_data[i];
          double p = 1;
          while (label_value >= 0) {
              p *= prob_data[(n * channels + label_value) * inner_num_ + s];
              label_value = parent_data[label_value];
          }
          top_data[(n * label_count + i) * inner_num_ + s] = static_cast<Dtype>(p);
      }

      // Find the argmax
#pragma omp parallel for
      for (int index = 0; index < outer_num_ * inner_num_; ++index) {
          // index == n * inner_num_ + s
          const int n = index / inner_num_;
          const int s = index % inner_num_;

          int argmax = 0;
          Dtype maxval = -FLT_MAX;
          for (int i = 0; i < label_count; ++i) {
              Dtype prob = top_data[(n * label_count + i) * inner_num_ + s];
              if (prob > maxval) {
                  argmax = label_data[i];
                  maxval = prob;
              }
          }

          argmax_data[n * inner_num_ + s] = argmax;
          if (max_data)
              max_data[n * inner_num_ + s] = maxval;
      }
      if (top.size() >= 2)
          top[1]->ShareData(prob_);
      return;
  }

  //---------------------------------------------------------------------------
  //                          Top Prediction
  //---------------------------------------------------------------------------
  auto child_data = tree_.child_.cpu_data();
  auto group_size_data = tree_.group_size_.cpu_data();
  auto group_offset_data = tree_.group_offset_.cpu_data();

#pragma omp parallel for
  for (int index = 0; index < outer_num_ * inner_num_; ++index) {
      // index == n * inner_num_ + s
      const int n = index / inner_num_;
      const int s = index % inner_num_;

      int g = 0; // start from the root
      double p = 1;
      int parent_argmax = 0;
      Dtype parent_p = 0;
      int argmax = 0;
      // Tree search
      do {
          auto offset = group_offset_data[g];
          auto size = group_size_data[g];
          Dtype maxval = -FLT_MAX;
          for (int j = 0; j < size; ++j) {
              Dtype prob = prob_data[(n * channels + offset + j) * inner_num_ + s];
              if (prob > maxval) {
                  argmax = offset + j;
                  maxval = prob;
              }
          }
          p *= maxval;
          g = child_data[argmax];
          if (p <= threshold_) {
              argmax = parent_argmax;
              p = parent_p;
              break;
          }
          parent_p = p;
          parent_argmax = argmax;
      } while (g > 0);

      top_data[n * inner_num_ + s] = static_cast<Dtype>(p);
      argmax_data[n * inner_num_ + s] = argmax;
  }
  if (top.size() >= 2)
      top[1]->ShareData(prob_);
}

#ifdef CPU_ONLY
STUB_GPU(TreePredictionLayer);
#endif

INSTANTIATE_CLASS(TreePredictionLayer);
REGISTER_LAYER_CLASS(TreePrediction);

}  // namespace caffe
