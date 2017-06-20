#include <algorithm>
#include <vector>
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "cuda.h"

#include "caffe/layers/reorg_layer.hpp"

namespace caffe {
#define BLOCK 512

dim3 cuda_gridsize(size_t n);

template <typename Dtype>
__global__ void reorg_kernel(int N, const Dtype *x, int w, int h, int c, int batch, int stride, int forward, Dtype *out)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int in_index = i;
    int in_w = i%w;
    i = i / w;
    int in_h = i%h;
    i = i / h;
    int in_c = i%c;
    i = i / c;
    int b = i%batch;

    int out_c = c / (stride*stride);

    int c2 = in_c % out_c;
    int offset = in_c / out_c;
    int w2 = in_w*stride + offset % stride;
    int h2 = in_h*stride + offset / stride;
    //printf("%d\n", offset);
    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));

    // printf("%d %d %d\n", w2, h2, c2);
    //printf("%d %d\n", in_index, out_index);
    //if(out_index >= N || out_index < 0) printf("bad bad bad \n");

    if (forward) out[out_index] = x[in_index];
    else out[in_index] = x[out_index];
    //if(forward) out[1] = x[1];
    //else out[0] = x[0];
}

template <typename Dtype>
__global__ void flatten_kernel(int N, const Dtype *x, int spatial, int layers, int batch, int forward, Dtype *out)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int in_s = i%spatial;
    i = i / spatial;
    int in_c = i%layers;
    i = i / layers;
    int b = i;

    int i1 = b*layers*spatial + in_c*spatial + in_s;
    int i2 = b*layers*spatial + in_s*layers + in_c;

    if (forward) out[i2] = x[i1];
    else out[i1] = x[i2];
}

template <typename Dtype>
void reorg_ongpu(const Dtype *x, int w, int h, int c, int batch, int stride, int forward, Dtype *out)
{
    int size = w*h*c*batch;
    reorg_kernel << <cuda_gridsize(size), BLOCK >> >(size, x, w, h, c, batch, stride, forward, out);
    CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void flatten_ongpu(const Dtype *x, int spatial, int layers, int batch, int forward, Dtype *out)
{
    int size = spatial*batch*layers;
    flatten_kernel << <cuda_gridsize(size), BLOCK >> >(size, x, spatial, layers, batch, forward, out);
    CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void ReorgLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const Dtype *bottom_data = bottom[0]->gpu_data();
    Dtype *top_data = top[0]->mutable_gpu_data();
    int num = bottom[0]->shape(0);
    int channels = bottom[0]->shape(1);
    int h = bottom[0]->shape(2);
    int w = bottom[0]->shape(3);

    if (flatten_) {
        if (reverse_) {
            flatten_ongpu(bottom_data, w*h, channels, num, 0, top_data);
        }
        else {
            flatten_ongpu(bottom_data, w*h, channels, num, 1, top_data);
        }
    }
    else if (reverse_) {
        reorg_ongpu(bottom_data, w, h, channels, num, stride_, 1, top_data);
    }
    else {
        reorg_ongpu(bottom_data, w, h, channels, num, stride_, 0, top_data);
    }
}

template <typename Dtype>
void ReorgLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype *top_diff = top[0]->gpu_diff();
    int num = top[0]->shape(0);
    int channels = top[0]->shape(1);
    int h = top[0]->shape(2);
    int w = top[0]->shape(3);

    if (flatten_) {
        if (reverse_) {
            flatten_ongpu(top_diff, w*h, channels, num, 1, bottom_diff);
        }
        else {
            flatten_ongpu(top_diff, w*h, channels, num, 0, bottom_diff);
        }
    }
    else if (reverse_) {
        reorg_ongpu(top_diff, w, h, channels, num, stride_, 0, bottom_diff);
    }
    else {
        reorg_ongpu(top_diff, w, h, channels, num, stride_, 1, bottom_diff);
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(ReorgLayer);
}  // namespace caffe
