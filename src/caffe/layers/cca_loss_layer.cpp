#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/cca_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

  template <typename Dtype>
  void CCALossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
  }

  template <typename Dtype>
  void CCALossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {
    // bottom = {prev_fc, fc_w, label}
    CHECK_EQ(bottom[0]->shape(0), bottom[2]->shape(0))
      << "The data and label should have the same first dimension.";

    vector<int> loss_shape(0); // Loss layers output a scalar; 0 axes.
    top[0]->Reshape(loss_shape);

    vector<int> temp_buffer_dims;
    temp_buffer_dims.push_back(bottom[0]->count() / bottom[0]->shape(0));
    temp_buffer_.Reshape(temp_buffer_dims);
  }

  template <typename Dtype>
  void CCALossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
  {
    Dtype cos_similar_loss = 0;
    const Dtype* weight_data = bottom[1]->cpu_data();
    const Dtype* feature_data = bottom[0]->cpu_data();
    int feature_dim = bottom[0]->count() / bottom[0]->shape(0);
    int weight_dim = bottom[1]->count() / bottom[1]->shape(0);
    CHECK_EQ(feature_dim, weight_dim);

    for (int i=0; i<bottom[0]->shape(0); i++) {
      int label = bottom[2]->cpu_data()[i];
      CHECK_LT(label, bottom[1]->shape(0));

      const Dtype* feature = feature_data + feature_dim * i;
      const Dtype* weight = weight_data + weight_dim * label;
      Dtype dotproduct = caffe_cpu_dot(feature_dim, feature, weight);
      Dtype feature_norm = sqrt(caffe_cpu_dot(feature_dim, feature, feature));
      Dtype weight_norm = sqrt(caffe_cpu_dot(weight_dim, weight, weight));
      CHECK_GT(feature_norm, FLT_MIN);
      CHECK_GT(weight_norm, FLT_MIN);

      cos_similar_loss += dotproduct / (feature_norm * weight_norm);
    }

    // Normalize the loss.
    top[0]->mutable_cpu_data()[0] = -cos_similar_loss / bottom[2]->count();
  }

  template <typename Dtype>
  void CCALossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
  {
    if (propagate_down[2]) {
      LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
    }

    if (propagate_down[0]) {
      const Dtype* weight_data = bottom[1]->cpu_data();
      const Dtype* feature_data = bottom[0]->cpu_data();
      int feature_dim = bottom[0]->count() / bottom[0]->shape(0);
      int weight_dim = bottom[1]->count() / bottom[1]->shape(0);
      CHECK_EQ(feature_dim, weight_dim);

      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      for (int i=0; i<bottom[0]->shape(0); i++) {
        int label = bottom[2]->cpu_data()[i];
        CHECK_LT(label, bottom[1]->shape(0));

        const Dtype* feature = feature_data + feature_dim * i;
        const Dtype* weight = weight_data + weight_dim * label;

        Dtype feature_norm = sqrt(caffe_cpu_dot(feature_dim, feature, feature));
        Dtype weight_norm = sqrt(caffe_cpu_dot(weight_dim, weight, weight));
        CHECK_GT(feature_norm, FLT_MIN);
        CHECK_GT(weight_norm, FLT_MIN);

        Dtype dotproduct = caffe_cpu_dot(feature_dim, feature, weight);
        Dtype cos_similarity = dotproduct / (feature_norm * weight_norm);
        Dtype* diff_buffer = bottom_diff + feature_dim * i;

        // (weight / weight_norm - feature/feature_norm * cos_similarity) / feature_norm
        caffe_copy(feature_dim, weight, diff_buffer);
        caffe_scal(feature_dim, static_cast<Dtype>(1.0)/weight_norm, diff_buffer);
        caffe_copy(feature_dim, feature, temp_buffer_.mutable_cpu_data());
        caffe_scal(feature_dim, static_cast<Dtype>(1.0)/feature_norm * cos_similarity, temp_buffer_.mutable_cpu_data());
        caffe_sub(feature_dim, diff_buffer, temp_buffer_.cpu_data(), diff_buffer);
        caffe_scal(feature_dim, static_cast<Dtype>(1.0)/feature_norm, diff_buffer);
      }

      Dtype loss_weight = -top[0]->cpu_diff()[0] / bottom[0]->shape(0);
      caffe_scal(bottom[0]->count(), loss_weight, bottom_diff);
    }
  }

#ifdef CPU_ONLY
  STUB_GPU(CCALossLayer);
#endif

  INSTANTIATE_CLASS(CCALossLayer);
  REGISTER_LAYER_CLASS(CCALoss);

} // namespace caffe
