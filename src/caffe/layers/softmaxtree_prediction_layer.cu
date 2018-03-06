#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmaxtree_prediction_layer.hpp"
#include "caffe/layers/softmaxtree_prediction_common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void kernel_smt_prediction(
    const int outer_num, const int root_size, const int inner_num,
    const TPredictTreeData<Dtype> tpd,
    Dtype* top_data) {
    CUDA_KERNEL_LOOP(index, outer_num * root_size * inner_num) {
        const int s = index % inner_num;
        const int g = (index / inner_num) % root_size;
        const int n = (index / inner_num) / root_size;

        predict_tree(tpd,
                     n, s, g,
                     top_data);
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
    auto root_size = tree_.root_size() + 1;

    caffe_gpu_set(top[0]->count(), Dtype(0), top_data);

    TPredictTreeData<Dtype> tpd(outer_num_, channels, inner_num_,
                                append_max_,
                                threshold_,
                                group_offset_data, group_size_data, child_data, child_size_data,
                                obj_data, prob_data);


    kernel_smt_prediction << <CAFFE_GET_BLOCKS(outer_num_ * root_size * inner_num_),
        CAFFE_CUDA_NUM_THREADS >> > (outer_num_, root_size, inner_num_,
                                     tpd,
                                     top_data);
    CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxTreePredictionLayer);


}  // namespace caffe
