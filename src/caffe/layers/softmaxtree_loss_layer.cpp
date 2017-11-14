#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmaxtree_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxTreeWithLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    LayerParameter softmaxtree_param(this->layer_param_);
    softmaxtree_param.clear_loss_weight(); // Ignore loss_wight if set
    softmaxtree_param.set_type("SoftmaxTree");
    softmaxtree_layer_ = boost::dynamic_pointer_cast<SoftmaxTreeLayer<Dtype>>(LayerRegistry<Dtype>::CreateLayer(softmaxtree_param));
    softmaxtree_bottom_vec_.clear();
    softmaxtree_bottom_vec_.push_back(bottom[0]);
    softmaxtree_top_vec_.clear();
    softmaxtree_top_vec_.push_back(&prob_);
    softmaxtree_layer_->SetUp(softmaxtree_bottom_vec_, softmaxtree_top_vec_);

    with_objectness_ = this->layer_param_.softmaxtree_loss_param().with_objectness();
    has_ignore_label_ = this->layer_param_.loss_param().has_ignore_label();
    if (has_ignore_label_) {
        ignore_label_ = this->layer_param_.loss_param().ignore_label();
        CHECK_LT(ignore_label_, 0) << "Ignore label must be negative";
    }
    if (!this->layer_param_.loss_param().has_normalization() &&
        this->layer_param_.loss_param().has_normalize()) {
        normalization_ = this->layer_param_.loss_param().normalize() ?
            LossParameter_NormalizationMode_VALID :
            LossParameter_NormalizationMode_BATCH_SIZE;
    } else {
        normalization_ = this->layer_param_.loss_param().normalization();
    }
#ifndef CPU_ONLY
    // Pre-fetch data
    if (Caffe::mode() == Caffe::GPU) {
        softmaxtree_layer_->softmax_tree_.parent_.mutable_gpu_data();
        softmaxtree_layer_->softmax_tree_.group_offset_.mutable_gpu_data();
        softmaxtree_layer_->softmax_tree_.group_size_.mutable_gpu_data();
        softmaxtree_layer_->softmax_tree_.group_.mutable_gpu_data();
    }
#endif
}

template <typename Dtype>
void SoftmaxTreeWithLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    LossLayer<Dtype>::Reshape(bottom, top);
    softmaxtree_layer_->Reshape(softmaxtree_bottom_vec_, softmaxtree_top_vec_);
    softmax_axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
    outer_num_ = bottom[0]->count(0, softmax_axis_);
    inner_num_ = bottom[0]->count(softmax_axis_ + 1);
    objectness_label_stride_ = inner_num_;
    if (with_objectness_ && bottom[1]->num_axes() < 3) {
        // Special case when we have N (or N*1) labels
        objectness_label_stride_ = 1;
        CHECK_EQ(outer_num_, bottom[1]->count())
            << "Number of labels must match number of predictions (or batches); "
            << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
            << "label count (number of labels) must be N*H*W (or N*1), "
            << "with integer values in {0, 1, ..., C-1}";
    } else {
        CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
            << "Number of labels must match number of predictions (or batches); "
            << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
            << "label count (number of labels) must be N*H*W (or N*1), "
            << "with integer values in {0, 1, ..., C-1}";
    }
    loss_.ReshapeLike(*bottom[1]);
    if (top.size() == 2) {
        if (with_objectness_)
            top[1]->Reshape({ outer_num_ , 1}); // index output
        else
            top[1]->ReshapeLike(*bottom[0]); // softmax output
    } else if (top.size() == 3) {
        assert(with_objectness_);

        top[1]->Reshape({ outer_num_ , 1 }); // index output
        top[2]->ReshapeLike(*bottom[0]); // softmax output
    }
    if (with_objectness_) {
        // TODO: add an image_axis_ (default 0) and add support for softmax_axis_ != 1
        CHECK_EQ(softmax_axis_, 1)
            << "Softmax axis must be 1 (other axes are not yet supported with objectness)";
        CHECK_EQ(outer_num_ * inner_num_, bottom[2]->count())
            << "Objects count must match number of predictions; "
            << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
            << "objects count (number of objectness predictions) must be N*H*W";

        label_prob_.Reshape({ outer_num_, inner_num_ });
        label_index_.Reshape({ outer_num_, 1 });
    }
}

template <typename Dtype>
Dtype SoftmaxTreeWithLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
    Dtype normalizer;
    switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
        normalizer = Dtype(outer_num_ * inner_num_);
        break;
    case LossParameter_NormalizationMode_VALID:
        if (valid_count == -1) {
            normalizer = Dtype(outer_num_ * inner_num_);
        } else {
            normalizer = Dtype(valid_count);
        }
        break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
        normalizer = Dtype(outer_num_);
        break;
    case LossParameter_NormalizationMode_NONE:
        normalizer = Dtype(1);
        break;
    default:
        LOG(FATAL) << "Unknown normalization mode: "
            << LossParameter_NormalizationMode_Name(normalization_mode);
    }
    // Some users will have no labels for some examples in order to 'turn off' a
    // particular loss in a multi-task setup. The max prevents NaNs in that case.
    return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void SoftmaxTreeWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    // The forward pass computes the softmax prob values.
    softmaxtree_layer_->Forward(softmaxtree_bottom_vec_, softmaxtree_top_vec_);
    auto prob_data = prob_.cpu_data();
    auto label = bottom[1]->cpu_data();
    auto parent_data = softmaxtree_layer_->softmax_tree_.parent_.cpu_data();
    int dim = prob_.count() / outer_num_;
    int nthreads = outer_num_ * inner_num_;
    auto loss_data = loss_.mutable_cpu_data();
    auto counts = loss_.mutable_cpu_diff();

    if (with_objectness_) {
        auto object_data = bottom[2]->cpu_data();
        auto label_prob_data = label_prob_.mutable_cpu_data();

#pragma omp parallel for
        for (int index = 0; index < nthreads; ++index) {
            // index == n * inner_num_ + s
            const int n = index / inner_num_;
            const int s = index % inner_num_;

            int label_value = static_cast<int>(label[n * objectness_label_stride_]);
            if (has_ignore_label_ && label_value == ignore_label_)
                continue;
            DCHECK_GE(label_value, 0);
            DCHECK_LT(label_value, prob_.shape(softmax_axis_));
            double p = object_data[index];  // Scale by objectness
            while (label_value >= 0) {
                p *= prob_data[n * dim + label_value * inner_num_ + s];
                label_value = parent_data[label_value];
            }

            label_prob_data[index] = p;
        }

        nthreads = outer_num_;
        auto label_index_data = label_index_.mutable_cpu_data();
#pragma omp parallel for
        for (int n = 0; n < nthreads; ++n) {
            const int label_value = static_cast<int>(label[n * objectness_label_stride_]);
            if (has_ignore_label_ && label_value == ignore_label_)
                continue;

            int max_idx = 0;
            double max_prob = 0;
            for (int j = 0; j < inner_num_; ++j) {
                if (label_prob_data[n * inner_num_ + j] > max_prob) {
                    max_prob = label_prob_data[n * inner_num_ + j];
                    max_idx = j;
                }
            }

            // This means index can go only as high as 24bits with float
            // Also note that index is over spatial dimension (not including C)
            label_index_data[n] = max_idx;
        }

#pragma omp parallel for
        for (int n = 0; n < nthreads; ++n) {
            loss_data[n] = 0;
            counts[n] = 0;

            int label_value = static_cast<int>(label[n * objectness_label_stride_]);
            if (has_ignore_label_ && label_value == ignore_label_)
                continue;

            int label_spatial_idx = static_cast<int>(label_index_data[n]);
            while (label_value >= 0) {
                loss_data[n] -= log(std::max(prob_data[n * dim + label_value * inner_num_ + label_spatial_idx], Dtype(FLT_MIN)));
                ++counts[n];
                label_value = parent_data[label_value];
            }
        }
    } else {
#pragma omp parallel for
        for (int index = 0; index < nthreads; ++index) {
            // index == n * inner_num_ + s
            const int n = index / inner_num_;
            const int s = index % inner_num_;
            loss_data[index] = 0;
            counts[index] = 0;
            int label_value = static_cast<int>(label[index]);
            if (has_ignore_label_ && label_value == ignore_label_)
                continue;

            DCHECK_GE(label_value, 0);
            DCHECK_LT(label_value, prob_.shape(softmax_axis_));
            while (label_value >= 0) {
                loss_data[index] -= log(std::max(prob_data[n * dim + label_value * inner_num_ + s], Dtype(FLT_MIN)));
                ++counts[index];
                label_value = parent_data[label_value];
            }
        }
    }
    Dtype loss = caffe_cpu_asum(nthreads, loss_data);
    int count = -1;
    // Only sum counts if we actually need the count of valid outputs.
    if (normalization_ == LossParameter_NormalizationMode_VALID) {
        count = caffe_cpu_asum(nthreads, counts);
        // Keep the count to re-use in backward
        counts[0] = count;
    }

    top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
    if (top.size() == 2) {
        if (with_objectness_)
            top[1]->ShareData(label_index_);
        else
            top[1]->ShareData(prob_);
    } else if (top.size() == 3) {
        assert(with_objectness_);

        top[1]->ShareData(label_index_);
        top[2]->ShareData(prob_);
    }
}

template <typename Dtype>
void SoftmaxTreeWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                   const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
        LOG(FATAL) << type()
            << " Layer cannot backpropagate to label inputs.";
    }
    if (with_objectness_) {
        if (propagate_down[2]) {
            LOG(FATAL) << type()
                << " Layer cannot backpropagate to objectness inputs.";
        }
    }
    if (!propagate_down[0])
        return;

    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    auto parent_data = softmaxtree_layer_->softmax_tree_.parent_.cpu_data();
    auto group_offset_data = softmaxtree_layer_->softmax_tree_.group_offset_.cpu_data();
    auto group_size_data = softmaxtree_layer_->softmax_tree_.group_size_.cpu_data();
    auto group_data = softmaxtree_layer_->softmax_tree_.group_.cpu_data();
    int dim = prob_.count() / outer_num_;
    int nthreads = outer_num_ * inner_num_;
    if (with_objectness_) {
        nthreads = outer_num_;
        auto label_index_data = label_index_.cpu_data();
#pragma omp parallel for
        for (int n = 0; n < nthreads; ++n) {
            int label_value = static_cast<int>(label[n * objectness_label_stride_]);
            if (has_ignore_label_ && label_value == ignore_label_)
                continue;
            int label_spatial_idx = label_index_data[n];
            while (label_value >= 0) {
                int g = group_data[label_value];
                int offset = group_offset_data[g];
                for (int c = 0; c < group_size_data[g]; ++c)
                    bottom_diff[n * dim + (offset + c) * inner_num_ + label_spatial_idx] = prob_data[n * dim + (offset + c) * inner_num_ + label_spatial_idx];

                bottom_diff[n * dim + label_value * inner_num_ + label_spatial_idx] -= 1;
                label_value = parent_data[label_value];
            }
        }
    } else {
#pragma omp parallel for
        for (int index = 0; index < nthreads; ++index) {
            // index == n * inner_num_ + s
            const int n = index / inner_num_;
            const int s = index % inner_num_;
            int label_value = static_cast<int>(label[index]);
            if (has_ignore_label_ && label_value == ignore_label_)
                continue;
            while (label_value >= 0) {
                int g = group_data[label_value];
                int offset = group_offset_data[g];
                for (int c = 0; c < group_size_data[g]; ++c)
                    bottom_diff[n * dim + (offset + c) * inner_num_ + s] = prob_data[n * dim + (offset + c) * inner_num_ + s];

                bottom_diff[n * dim + label_value * inner_num_ + s] -= 1;
                label_value = parent_data[label_value];
            }
        }
    }
    int count = -1;
    if (normalization_ == LossParameter_NormalizationMode_VALID) {
        count = loss_.cpu_diff()[0];
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / get_normalizer(normalization_, count);
    caffe_scal(bottom[0]->count(), loss_weight, bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxTreeWithLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxTreeWithLossLayer);
REGISTER_LAYER_CLASS(SoftmaxTreeWithLoss);

}  // namespace caffe
