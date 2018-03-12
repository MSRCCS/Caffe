#include <algorithm>
#include <vector>
#include <float.h>
#include <stack>

#include "caffe/layers/softmaxtree_prediction_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

struct Pred {
    double parent_p;
    int parent_argmax;
    int g; 
};

template <typename Dtype>
void predict_tree_stack(int outer_num, int channels, int inner_num,
                        bool append_max,
                        float threshold,
                        const int* group_offset_data, const int* group_size_data, const int* child_data, const int* child_size_data,
                        const Dtype* obj_data, const Dtype* prob_data,
                        int max_stack_size, int n, int s, int g,
                        Dtype* top_data) {
    std::stack<Pred> preds;
    preds.push({ 1.0, -1, g });
    while (!preds.empty()) {
        DCHECK_LE(preds.size(), max_stack_size);
        auto pred = preds.top();
        preds.pop();
        double p = pred.parent_p;
        int argmax = 0;
        {
            g = pred.g;
            Dtype maxval = -FLT_MAX;
            auto offset = group_offset_data[g];
            auto size = group_size_data[g];
            for (int j = 0; j < size; ++j) {
                Dtype prob = prob_data[(n * channels + offset + j) * inner_num + s];
                if (prob > maxval) {
                    argmax = offset + j;
                    maxval = prob;
                }
            }
            p *= maxval;
        }
        if (p > threshold) {
            g = child_data[argmax]; // initial child group
            if (g >= 0) {
                // if there is any child, descend further
                int sg_count = child_size_data[argmax] + 1;
                for (int sg = 0; sg < sg_count; ++sg)
                    preds.push({ p, argmax, g + sg });
                continue;
            }
        } else {
            argmax = pred.parent_argmax;
            if (argmax < 0)
                continue;
            p = pred.parent_p;
        }

        const int top_channels = append_max ? channels + 1 : channels;
        Dtype node_p = obj_data ? obj_data[n * inner_num + s] : static_cast<Dtype>(p);
        top_data[(n * top_channels + argmax) * inner_num + s] = node_p;
        if (append_max) {
            int max_idx = (n * top_channels + channels) * inner_num + s;
            if (node_p > top_data[max_idx])
                top_data[max_idx] = node_p;
        }
    }
}

template <typename Dtype>
void SoftmaxTreePredictionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    Layer<Dtype>::LayerSetUp(bottom, top);
    tree_.read(this->layer_param().softmaxtreeprediction_param().tree().c_str());
    threshold_ = this->layer_param().softmaxtreeprediction_param().threshold();
    append_max_ = this->layer_param().softmaxtreeprediction_param().append_max();
    with_objectness_ = bottom.size() == 2;

    stack_size_ = 0;
    auto root_size = tree_.root_size() + 1;
    for (int g = 0; g < root_size; ++g) {
        int stack_size = find_max_stack_size(g);
        if (stack_size > stack_size_)
            stack_size_ = stack_size;
    }

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

#ifndef CPU_ONLY
    if (Caffe::mode() == Caffe::GPU) {
        shape = { outer_num_, tree_.root_size() + 1, inner_num_, stack_size_ };
        stack_parent_p_.Reshape(shape);
        stack_parent_argmax_.Reshape(shape);
        stack_g_.Reshape(shape);
        stack_parent_p_.mutable_gpu_data();
        stack_parent_argmax_.mutable_gpu_data();
        stack_g_.mutable_gpu_data();
    }
#endif

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

#pragma omp parallel for
    for (int index = 0; index < outer_num_ * root_size * inner_num_; ++index) {
        const int s = index % inner_num_;
        const int g = (index / inner_num_) % root_size;
        const int n = (index / inner_num_) / root_size;
        
        predict_tree_stack(outer_num_, channels, inner_num_,
                           append_max_,
                           threshold_,
                           group_offset_data, group_size_data, child_data, child_size_data,
                           obj_data, prob_data,
                           stack_size_, n, s, g,
                           top_data);
    }
}

template <typename Dtype>
int SoftmaxTreePredictionLayer<Dtype>::find_max_stack_size(int g) {
    auto child_data = tree_.child_.cpu_data();
    auto child_size_data = tree_.child_size_.cpu_data();
    auto group_size_data = tree_.group_size_.cpu_data();
    auto group_offset_data = tree_.group_offset_.cpu_data();
    int max_stack_size = 1;

    auto offset = group_offset_data[g];
    auto size = group_size_data[g];

    for (int n = offset; n < offset + size; ++n) {
        g = child_data[n]; // initial child group
        if (g < 0)
            continue;
        int stack_size = child_size_data[n] + this->find_max_stack_size(g);
        if (stack_size > max_stack_size)
            max_stack_size = stack_size;
    }

    return max_stack_size;
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxTreePredictionLayer);
#endif

INSTANTIATE_CLASS(SoftmaxTreePredictionLayer);
REGISTER_LAYER_CLASS(SoftmaxTreePrediction);

}  // namespace caffe
