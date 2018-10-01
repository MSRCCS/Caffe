#include <algorithm>
#include <vector>
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "cuda.h"

#include "caffe/layers/region_prediction_layer.hpp"

namespace caffe {

template <typename Dtype>
void RegionPredictionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    this->Forward_cpu(bottom, top);
}

template <typename Dtype>
void RegionPredictionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); i++) {
        CHECK(!propagate_down[i]);
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(RegionPredictionLayer);

}  // namespace caffe

