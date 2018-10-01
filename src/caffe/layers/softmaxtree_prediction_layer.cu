#include <algorithm>
#include <cfloat>
#include <vector>
#include <assert.h>

#include "caffe/layers/softmaxtree_prediction_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

__device__ void stack_push(double* parent_p_data, int* parent_argmax_data, int* g_data,
                           int& stack_size,
                           double p, int argmax, int g) {
    parent_p_data[stack_size] = p;
    parent_argmax_data[stack_size] = argmax;
    g_data[stack_size] = g;
    stack_size++;
}

__device__ void stack_pop(const double* parent_p_data, const int* parent_argmax_data, const int* g_data,
                          int& stack_size,
                          double& p, int& argmax, int& g) {
    assert(stack_size > 0);
    stack_size--;
    p = parent_p_data[stack_size];
    argmax = parent_argmax_data[stack_size];
    g = g_data[stack_size];
}

template <typename Dtype>
__device__ void predict_tree_stack(
    int outer_num, int channels, int inner_num,
    bool append_max,
    float threshold,
    const int* group_offset_data, const int* group_size_data, const int* child_data, const int* child_size_data,
    double* parent_p_data, int* parent_argmax_data, int* g_data,
    const Dtype* obj_data, const Dtype* prob_data,
    int max_stack_size, int n, int s, int g,
    Dtype* top_data,
    bool output_tree_path) {

    int stack_size = 0;
    const int top_channels = append_max ? (channels + 1) : channels;
    Dtype obj = obj_data ? obj_data[n * inner_num + s] : 1;
    double root_p = output_tree_path ? obj : 1.0;
    threshold = output_tree_path ? (threshold * obj) : threshold;
    stack_push(parent_p_data, parent_argmax_data, g_data,
               stack_size,
               root_p, -1, g);
    while (stack_size) {
        assert(stack_size <= max_stack_size);
        double parent_p;
        int parent_argmax;
        int g;
        stack_pop(parent_p_data, parent_argmax_data, g_data,
                  stack_size,
                  parent_p, parent_argmax, g);
        double p = parent_p;
        int argmax = 0;
        {
            Dtype maxval = -FLT_MAX;
            auto offset = group_offset_data[g];
            argmax = offset;
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
            if (output_tree_path) {
                top_data[(n * top_channels + argmax) * inner_num + s] = static_cast<Dtype>(p);
            }
            g = child_data[argmax]; // initial child group
            if (g >= 0) {
                // if there is any child, descend further
                int sg_count = child_size_data[argmax] + 1;
                for (int sg = 0; sg < sg_count; ++sg) {
                    stack_push(parent_p_data, parent_argmax_data, g_data,
                               stack_size,
                               p, argmax, g + sg);

                }
                continue;
            }
        } else {
            argmax = parent_argmax;
            if (argmax < 0)
                continue;
            p = parent_p;
        }
        
        Dtype node_p = 0;
        if (!output_tree_path) {
            node_p = obj_data ? obj : static_cast<Dtype>(p);
            top_data[(n * top_channels + argmax) * inner_num + s] = node_p;
        }
        if (append_max) {
            int max_idx = (n * top_channels + channels) * inner_num + s;
            if (output_tree_path) {
                // in this case, we use the obj as the max value, which will be
                // used as the indicator for class-independent NMS. or the
                // maximum value will always be the ones in the root.
                // gradually, we might remove the support of append_max since
                // it is more like a legacy strategy
                top_data[max_idx] = obj;
            } else {
                if (node_p > top_data[max_idx]) {
                    top_data[max_idx] = node_p;
                }
            }
        }
    }
}

template <typename Dtype>
__global__ void kernel_smt_prediction(
    int outer_num, int channels, int inner_num, int root_size,
    bool append_max,
    float threshold,
    const int* group_offset_data, const int* group_size_data, const int* child_data, const int* child_size_data,
    double* parent_p_data, int* parent_argmax_data, int* g_data,
    const Dtype* obj_data, const Dtype* prob_data,
    int max_stack_size,
    Dtype* top_data,
    bool output_tree_path) {
    CUDA_KERNEL_LOOP(index, outer_num * root_size * inner_num) {
        const int s = index % inner_num;
        const int g = (index / inner_num) % root_size;
        const int n = (index / inner_num) / root_size;

        predict_tree_stack(outer_num, channels, inner_num,
                           append_max,
                           threshold,
                           group_offset_data, group_size_data, child_data, child_size_data,
                           &parent_p_data[index * max_stack_size], &parent_argmax_data[index * max_stack_size], &g_data[index * max_stack_size],
                           obj_data, prob_data,
                           max_stack_size, n, s, g,
                           top_data, 
                           output_tree_path);
    }
}

template <typename Dtype>
void SoftmaxTreePredictionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {

    auto top_data = top[0]->mutable_gpu_data();
    auto prob_data = bottom[0]->gpu_data();
    const Dtype* obj_data = with_objectness_ ? bottom[1]->gpu_data() : NULL;
    int channels = bottom[0]->shape(axis_);

    auto child_data = tree_.child_.gpu_data();
    auto child_size_data = tree_.child_size_.gpu_data();
    auto group_size_data = tree_.group_size_.gpu_data();
    auto group_offset_data = tree_.group_offset_.gpu_data();
    auto parent_p_data = stack_parent_p_.mutable_gpu_data();
    auto parent_argmax_data = stack_parent_argmax_.mutable_gpu_data();
    auto g_data = stack_g_.mutable_gpu_data();
    auto root_size = tree_.root_size() + 1;

    caffe_gpu_set(top[0]->count(), Dtype(0), top_data);

    kernel_smt_prediction << <CAFFE_GET_BLOCKS(outer_num_ * root_size * inner_num_),
        CAFFE_CUDA_NUM_THREADS >> > (outer_num_, channels, inner_num_, root_size,
                                     append_max_,
                                     threshold_,
                                     group_offset_data, group_size_data, child_data, child_size_data,
                                     parent_p_data, parent_argmax_data, g_data,
                                     obj_data, prob_data,
                                     stack_size_,
                                     top_data,
                                     output_tree_path_);
    CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxTreePredictionLayer);


}  // namespace caffe
