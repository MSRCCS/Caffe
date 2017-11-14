#include <vector>

#include "caffe/layers/indexed_threshold_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void IndexedThresholdLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    null_scale_ = this->layer_param_.indexedthreshold_loss_param().null_scale();
    positive_scale_ = this->layer_param_.indexedthreshold_loss_param().positive_scale();
    threshold_ = this->layer_param_.indexedthreshold_loss_param().threshold();
}

template <typename Dtype>
void IndexedThresholdLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::Reshape(bottom, top);
    index_axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.indexedthreshold_loss_param().axis());
    outer_num_ = bottom[0]->count(0, index_axis_ + 1);
    CHECK_EQ(outer_num_, bottom[1]->count())
        << "Index inputs size must be equal the index axis outer size.";
    diff_.ReshapeLike(*bottom[0]);
    weights_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void IndexedThresholdLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                   const vector<Blob<Dtype>*>& top) {
    int count = bottom[0]->count();
    auto index_data = bottom[1]->cpu_data();
    auto bottom_data = bottom[0]->cpu_data();
    auto diff_data = diff_.mutable_cpu_data();
    auto weights_data = weights_.mutable_cpu_data();
    auto dim = bottom[0]->count(index_axis_ + 1);
    const Dtype alpha = sqrt(null_scale_);
    caffe_set(count, alpha, weights_data);
    if (null_scale_ == 1) {
        caffe_copy(count, bottom_data, diff_data);
    } else {
        caffe_axpy(
            count,       // count
            alpha,       // alpha
            bottom_data, // a
            diff_data);  // b
    }

    Dtype scale = sqrt(positive_scale_);
#pragma omp parallel for
    for (int i = 0; i < outer_num_; ++i) {
        auto index = static_cast<int>(index_data[i]);
        CHECK_LT(index, dim) << "Positive target index out of range";

        if (bottom_data[dim * i + index] < threshold_) {
            diff_data[dim * i + index] = scale * (bottom_data[dim * i + index] - threshold_);
            weights_data[dim * i + index] = scale;
        } else {
            diff_data[dim * i + index] = 0;
        }
    }
    Dtype dot = 0;
    dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
    Dtype loss = dot / bottom[0]->num() / Dtype(2);
    top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void IndexedThresholdLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
        LOG(FATAL) << type()
            << " Layer cannot backpropagate to index inputs.";
    }
    if (!propagate_down[0])
        return;

    int count = bottom[0]->count();
    caffe_mul(count, weights_.cpu_data(), diff_.cpu_data(), diff_.mutable_cpu_data());
    // Scale gradiant by loss
    const Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num();
    caffe_axpy(
        count,                           // count
        alpha,                           // alpha
        diff_.cpu_data(),                // a
        bottom[0]->mutable_cpu_diff());  // b
}

#ifdef CPU_ONLY
STUB_GPU(IndexedThresholdLossLayer);
#endif

INSTANTIATE_CLASS(IndexedThresholdLossLayer);
REGISTER_LAYER_CLASS(IndexedThresholdLoss);

}  // namespace caffe
