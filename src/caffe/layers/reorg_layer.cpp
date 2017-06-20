#include <vector>

#include <stdlib.h>
#include "caffe/layers/reorg_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void reorg_cpu(const Dtype *x, int w, int h, int c, int batch, int stride, int forward, Dtype *out)
{
    int b,i,j,k;
    int out_c = c/(stride*stride);

    for(b = 0; b < batch; ++b){
        for(k = 0; k < c; ++k){
            for(j = 0; j < h; ++j){
                for(i = 0; i < w; ++i){
                    int in_index  = i + w*(j + h*(k + c*b));
                    int c2 = k % out_c;
                    int offset = k / out_c;
                    int w2 = i*stride + offset % stride;
                    int h2 = j*stride + offset / stride;
                    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));
                    if(forward) out[out_index] = x[in_index];
                    else out[in_index] = x[out_index];
                }
            }
        }
    }
}

template <typename Dtype>
void flatten(Dtype *x, int size, int layers, int batch, int forward)
{
    vector<Dtype> swap(size*layers*batch);
    int i,c,b;
    for(b = 0; b < batch; ++b){
        for(c = 0; c < layers; ++c){
            for(i = 0; i < size; ++i){
                int i1 = b*layers*size + c*size + i;
                int i2 = b*layers*size + i*layers + c;
                if (forward) swap[i2] = x[i1];
                else swap[i1] = x[i2];
            }
        }
    }
    caffe_copy(size*layers*batch, &swap[0], x);
}

template <typename Dtype>
void ReorgLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    Layer<Dtype>::LayerSetUp(bottom, top);
    const ReorgParameter &reorg_param = this->layer_param().reorg_param();
    stride_ = reorg_param.stride();
    reverse_ = reorg_param.reverse();
    flatten_ = reorg_param.flatten();
}

template <typename Dtype>
void ReorgLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    vector<int> shape = bottom[0]->shape();
    if (reverse_) {
        shape[1] /= (stride_ * stride_);
        shape[2] *= stride_;
        shape[3] *= stride_;
    }
    else {
        shape[1] *= (stride_ * stride_);
        shape[2] /= stride_;
        shape[3] /= stride_;
    }
    top[0]->Reshape(shape);
}

template <typename Dtype>
void ReorgLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const Dtype *bottom_data = bottom[0]->cpu_data();
    Dtype *top_data = top[0]->mutable_cpu_data();
    int num = bottom[0]->shape(0);
    int channels = bottom[0]->shape(1);
    int h = bottom[0]->shape(2);
    int w = bottom[0]->shape(3);

    if (flatten_) {
        caffe_copy(bottom[0]->count(), bottom_data, top_data);
        if (reverse_) {
            flatten(top_data, w*h, channels, num, 0);
        }
        else {
            flatten(top_data, w*h, channels, num, 1);
        }
    }
    else if (reverse_) {
        reorg_cpu(bottom_data, w, h, channels, num, stride_, 1, top_data);
    }
    else {
        reorg_cpu(bottom_data, w, h, channels, num, stride_, 0, top_data);
    }
}

template <typename Dtype>
void ReorgLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype *top_diff = top[0]->cpu_diff();
    int num = top[0]->shape(0);
    int channels = top[0]->shape(1);
    int h = top[0]->shape(2);
    int w = top[0]->shape(3);

    if (flatten_) {
        caffe_copy(top[0]->count(), top_diff, bottom_diff);
        if (reverse_) {
            flatten(bottom_diff, w*h, channels, num, 1);
        }
        else {
            flatten(bottom_diff, w*h, channels, num, 0);
        }
    }
    else if (reverse_) {
        reorg_cpu(top_diff, w, h, channels, num, stride_, 0, bottom_diff);
    }
    else {
        reorg_cpu(top_diff, w, h, channels, num, stride_, 1, bottom_diff);
    }
}

#ifdef CPU_ONLY
STUB_GPU(ReorgLayer);
#endif

INSTANTIATE_CLASS(ReorgLayer);
REGISTER_LAYER_CLASS(Reorg);

}  // namespace caffe
