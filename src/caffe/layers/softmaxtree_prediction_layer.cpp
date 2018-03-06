#include <algorithm>
#include <vector>
#include <float.h>
#include <stack>

#include "caffe/layers/softmaxtree_prediction_layer.hpp"
#include "caffe/layers/softmaxtree_prediction_common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxTreePredictionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    Layer<Dtype>::LayerSetUp(bottom, top);
    tree_.read(this->layer_param().softmaxtreeprediction_param().tree().c_str());
    threshold_ = this->layer_param().softmaxtreeprediction_param().threshold();
    append_max_ = this->layer_param().softmaxtreeprediction_param().append_max();
    with_objectness_ = bottom.size() == 2;

#ifndef CPU_ONLY
    // Pre-fetch data
    if (Caffe::mode() == Caffe::GPU) {
        tree_.group_size_.mutable_gpu_data();
        tree_.group_offset_.mutable_gpu_data();
        tree_.child_.mutable_gpu_data();
        tree_.child_size_.mutable_gpu_data();
    }
#endif
}

template <typename Dtype>
void SoftmaxTreePredictionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                const vector<Blob<Dtype>*>& top) {
    axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
    CHECK_GE(bottom[0]->num_axes(), 4);
    int channels = bottom[0]->shape(axis_);
    CHECK(channels == tree_.nodes()) << "Channel count: " << channels << " must match tree node count: " << tree_.nodes();
    outer_num_ = bottom[0]->count(0, axis_);
    inner_num_ = bottom[0]->count(axis_ + 1);

    auto shape = bottom[0]->shape();
    if (append_max_)
        shape[axis_] = channels + 1;
    top[0]->Reshape(shape);

    // if objectness is provided
    if (with_objectness_)
        CHECK_EQ(bottom[0]->count() / channels, bottom[1]->count());
}

template <typename Dtype>
void SoftmaxTreePredictionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    auto top_data = top[0]->mutable_cpu_data();
    auto prob_data = bottom[0]->cpu_data();
    const Dtype* obj_data = with_objectness_ ? bottom[1]->cpu_data() : NULL;
    int channels = bottom[0]->shape(axis_);

    auto child_data = tree_.child_.cpu_data();
    auto child_size_data = tree_.child_size_.cpu_data();
    auto group_size_data = tree_.group_size_.cpu_data();
    auto group_offset_data = tree_.group_offset_.cpu_data();
    auto root_size = tree_.root_size() + 1;

    caffe_set(top[0]->count(), Dtype(0), top_data);

    TPredictTreeData<Dtype> tpd(outer_num_, channels, inner_num_,
                                append_max_,
                                threshold_,
                                group_offset_data, group_size_data, child_data, child_size_data,
                                obj_data, prob_data);
#pragma omp parallel for
    for (int index = 0; index < outer_num_ * root_size * inner_num_; ++index) {
        const int s = index % inner_num_;
        const int g = (index / inner_num_) % root_size;
        const int n = (index / inner_num_) / root_size;
        
        predict_tree(tpd,
                     n, s, g,
                     top_data);
    }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxTreePredictionLayer);
#endif

INSTANTIATE_CLASS(SoftmaxTreePredictionLayer);
REGISTER_LAYER_CLASS(SoftmaxTreePrediction);

}  // namespace caffe
