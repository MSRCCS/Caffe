#include <algorithm>
#include <vector>
#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include "cuda.h"

#include "caffe/layers/region_loss_layer.hpp"

namespace caffe {
#define BLOCK 512

dim3 cuda_gridsize(size_t n){
    size_t k = (n-1) / BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
    }
    dim3 d;
    d.x = (unsigned int)x;
    d.y = (unsigned int)y;
    d.z = 1;

    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
    return d;
}

__device__ void softmax_device(const float *input, int n, float temp, int stride, float *output)
{
    int i;
    float sum = 0;
    float largest = -INFINITY;
    for(i = 0; i < n; ++i){
        int val = input[i*stride];
        largest = (val>largest) ? val : largest;
    }
    for(i = 0; i < n; ++i){
        float e = exp(input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}

__global__ void softmax_kernel(const float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch*groups) return;
    int b = id / groups;
    int g = id % groups;
    softmax_device(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
}

void softmax_gpu(const float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output)
{
    softmax_kernel<<<cuda_gridsize(batch*groups), BLOCK>>>(input, n, batch, batch_offset, groups, group_offset, stride, temp, output);
    CUDA_POST_KERNEL_CHECK;
}

__device__ float logistic_activate_kernel(float x) { return 1. / (1. + exp(-x)); }
__device__ float logistic_gradient_kernel(float x) { return (1 - x)*x; }

__device__ float activate_kernel(float x, ACTIVATION a)
{
    switch (a) {
    case LOGISTIC:
        return logistic_activate_kernel(x);
    }
    return 0;
}

__device__ float gradient_kernel(float x, ACTIVATION a)
{
    switch (a) {
    case LOGISTIC:
        return logistic_gradient_kernel(x);
    }
    return 0;
}

__global__ void activate_array_kernel(float *x, int n, ACTIVATION a)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) x[i] = activate_kernel(x[i], a);
}

__global__ void gradient_array_kernel(float *x, int n, ACTIVATION a, float *delta)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < n) delta[i] *= gradient_kernel(x[i], a);
}

void activate_array_ongpu(float *x, int n, ACTIVATION a)
{
    activate_array_kernel << <cuda_gridsize(n), BLOCK >> >(x, n, a);
    CUDA_POST_KERNEL_CHECK;
}

void gradient_array_ongpu(float *x, int n, ACTIVATION a, float *delta)
{
    gradient_array_kernel << <cuda_gridsize(n), BLOCK >> >(x, n, a, delta);
    CUDA_POST_KERNEL_CHECK;
}

__global__ void axpy_kernel(int N, float ALPHA, float *X, int OFFX, int INCX, float *Y, int OFFY, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i < N) Y[OFFY + i*INCY] += ALPHA*X[OFFX + i*INCX];
}

void axpy_ongpu_offset(int N, float ALPHA, float * X, int OFFX, int INCX, float * Y, int OFFY, int INCY)
{
    axpy_kernel << <cuda_gridsize(N), BLOCK >> >(N, ALPHA, X, OFFX, INCX, Y, OFFY, INCY);
    CUDA_POST_KERNEL_CHECK;
}

void axpy_ongpu(int N, float ALPHA, float * X, int INCX, float * Y, int INCY)
{
    axpy_ongpu_offset(N, ALPHA, X, 0, INCX, Y, 0, INCY);
}

int *cuda_make_int_array(int *x, size_t n)
{
    int *x_gpu;
    size_t size = sizeof(int)*n;
    CUDA_CHECK(cudaMalloc((void **)&x_gpu, size));
    if(x){
        CUDA_CHECK(cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice));
    }
    return x_gpu;
}

__global__ void softmax_tree_kernel(const float *input, int spatial, int batch, int stride, float temp, float *output, int groups, int *group_size, int *group_offset)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= spatial*batch*groups) return;
    int s = id % spatial;
    id = id / spatial;
    int g = id % groups;
    int b = id / groups;
    int goff = group_offset[g]*spatial;
    int boff = b*stride;
    softmax_device(input + goff + boff + s, group_size[g], temp, spatial, output + goff + boff + s);
}

void cuda_free(float *x_gpu)
{
    CUDA_CHECK(cudaFree(x_gpu));
}

void softmax_tree(const float *input, int spatial, int batch, int stride, float temp, float *output, tree &hier)
{
    if (hier.group_size_gpu == NULL) {
        hier.group_size_gpu = cuda_make_int_array(hier.group_size, hier.groups);
        hier.group_offset_gpu = cuda_make_int_array(hier.group_offset, hier.groups);
    } 
    int* tree_groups_size = hier.group_size_gpu;
    int* tree_groups_offset = hier.group_offset_gpu;
    int num = spatial*batch*hier.groups;
    softmax_tree_kernel<<<cuda_gridsize(num), BLOCK>>>(input, spatial, batch, stride, temp, output, hier.groups, tree_groups_size, tree_groups_offset);
    CUDA_CHECK(cudaPeekAtLastError());
}

template <typename Dtype>
void RegionLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    // prepare wrapper environment for forward computation
    network &net = this->net_;
    layer &l = this->l_;
    prepare_net_layer(net, l, bottom, top);
    const float* input_gpu = (const float *)bottom[0]->gpu_data();
    l.output_gpu = (float*) output_gpu_.mutable_gpu_data();
    
    // perform computation
    caffe_gpu_memcpy(l.outputs * l.batch * sizeof(float), input_gpu, l.output_gpu);
    for (int b = 0; b < l.batch; ++b) {
        for (int n = 0; n < l.n; ++n) {
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array_ongpu(l.output_gpu + index, 2 * l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, l.coords);
            activate_array_ongpu(l.output_gpu + index, l.w*l.h, LOGISTIC);
        }
    }

    if (l.softmax_tree) {
        int index = entry_index(l, 0, 0, l.coords + 1);
        softmax_tree(input_gpu + index, l.w*l.h, l.batch*l.n, l.inputs/l.n, l.temperature, l.output_gpu + index, *l.softmax_tree);
    }
    else if (l.softmax) {
        int index = entry_index(l, 0, 0, l.coords + 1);
        softmax_gpu(input_gpu + index, l.classes, l.batch*l.n, l.inputs / l.n, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu + index);
    }

    Dtype* dst = output_.mutable_cpu_data();
    caffe_gpu_memcpy(output_.count() * sizeof(Dtype), l.output_gpu, dst);

    l.output = (float *)output_.mutable_cpu_data();
    l.delta = (float *)output_.mutable_cpu_diff();
    forward_for_loss(net, l);
   
    Dtype* output_gpu_diff = output_gpu_.mutable_gpu_diff();
    caffe_gpu_memcpy(output_gpu_.count() * sizeof(Dtype), l.delta, output_gpu_diff);
    Dtype dot;
    caffe_gpu_dot(output_gpu_.count(), output_gpu_diff, output_gpu_diff, &dot);
    *l.cost = dot;
    // multiplicate delta with -loss_weight to fit for caffe's sgd solver.
    caffe_gpu_scale(output_gpu_.count(), (Dtype)-l.loss_weight / l.batch, output_gpu_diff, output_gpu_diff);
    *(l.cost) /= l.batch;
}

template <typename Dtype>
void RegionLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    // prepare wrapper environment for forward computation
    network &net = this->net_;
    layer &l = this->l_;
    net.input = NULL;
    prepare_net_layer(net, l, bottom, top);
    float* output_gpu = (float *)output_gpu_.gpu_data();
    l.delta_gpu = (float *)bottom[0]->mutable_gpu_diff();
    /*caffe_gpu_memcpy(l.outputs * l.batch * sizeof(Dtype), output_.gpu_diff(), l.delta_gpu);*/
    caffe_gpu_memcpy(l.outputs * l.batch * sizeof(Dtype), output_gpu_.gpu_diff(), l.delta_gpu);

    for (int b = 0; b < l.batch; ++b) {
        for (int n = 0; n < l.n; ++n) {
            int index = entry_index(l, b, n*l.w*l.h, 0);
            gradient_array_ongpu(output_gpu + index, 2 * l.w*l.h, LOGISTIC, l.delta_gpu + index);
            index = entry_index(l, b, n*l.w*l.h, l.coords);
            gradient_array_ongpu(output_gpu + index, l.w*l.h, LOGISTIC, l.delta_gpu + index);
        }
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(RegionLossLayer);

template <typename Dtype>
void RegionOutputLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    layer l = this->l_;
    l.output_gpu = (float *)output_gpu_.mutable_gpu_data();
    network net;
    net.input_gpu = (float *)bottom[0]->gpu_data();

    // As in RegionLossLayer, apply sigmoid or softmax operations on bottom data.
    caffe_gpu_memcpy(l.outputs * l.batch * sizeof(float), net.input_gpu, l.output_gpu);
    for (int b = 0; b < l.batch; ++b) {
        for (int n = 0; n < l.n; ++n) {
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array_ongpu(l.output_gpu + index, 2 * l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, l.coords);
            activate_array_ongpu(l.output_gpu + index, l.w*l.h, LOGISTIC);
        }
    }
    if (l.softmax_tree) {
        // int count = 5;
        // for (int i = 0; i < l.softmax_tree->groups; ++i) {
        //     int group_size = l.softmax_tree->group_size[i];
        //     int index = entry_index(l, 0, 0, count);
        //     softmax_gpu(net.input_gpu + index, group_size, l.batch*l.n, l.inputs / l.n, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu + index);
        //     count += group_size;
        // }
        int index = entry_index(l, 0, 0, l.coords + 1);
        softmax_tree(net.input_gpu + index, l.w*l.h, l.batch*l.n, l.inputs/l.n, l.temperature, l.output_gpu + index, *l.softmax_tree);
    }
    else if (l.softmax) {
        int index = entry_index(l, 0, 0, l.coords + 1);
        //printf("%d\n", index);
        softmax_gpu(net.input_gpu + index, l.classes, l.batch*l.n, l.inputs / l.n, l.w*l.h, 1, l.w*l.h, 1, l.output_gpu + index);
    }

    Dtype* dst = output_.mutable_cpu_data();
    caffe_gpu_memcpy(output_.count() * sizeof(Dtype), l.output_gpu, dst);
    GetRegionBoxes(bottom, top);
}

INSTANTIATE_LAYER_GPU_FUNCS(RegionOutputLayer);
}  // namespace caffe
