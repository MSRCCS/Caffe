#include <algorithm>
#include <vector>
#include <cfloat>

#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
__global__ void channel_div_kernel(const int n, const int channel, const int spat_dim, const Dtype* a,
  const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    int ch_res = index / spat_dim;
    int ch_idx = ch_res % channel;
    y[index] = a[index] / (b[ch_idx] + FLT_EPSILON);
  }
}

template <typename Dtype>
__global__ void channel_div_kernel_neps(const int n, const int channel, const int spat_dim, const Dtype* a,
  const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    int ch_res = index / spat_dim;
    int ch_idx = ch_res % channel;
    y[index] = a[index] / (b[ch_idx]);
  }
}

template <typename Dtype>
__global__ void channel_sub_kernel(const int n, const int channel, const int spat_dim, const Dtype* a,
  const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    int ch_res = index / spat_dim;
    int ch_idx = ch_res % channel;
    y[index] = a[index] - b[ch_idx];
  }
}

template <typename Dtype>
__global__ void num_mul_kernel(const int n, const int channel, const int spat_dim, const Dtype* a,
  const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    int ch_res = index / spat_dim;
    int num_idx = ch_res / channel;
    y[index] = a[index] * b[num_idx];
  }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int num = bottom[0]->shape(0);
  int spatial_dim = bottom[0]->count()/(channels_*bottom[0]->shape(0));

  if (bottom[0] != top[0]) {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }


  if (use_global_stats_) {
    // use the stored mean/variance estimates.
    const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
        0 : 1 / this->blobs_[2]->cpu_data()[0];
    caffe_gpu_scale(variance_.count(), scale_factor,
        this->blobs_[0]->gpu_data(), mean_.mutable_gpu_data());
    caffe_gpu_scale(variance_.count(), scale_factor,
        this->blobs_[1]->gpu_data(), variance_.mutable_gpu_data());
  } else {
    // compute mean
    caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
        1. / (num * spatial_dim), bottom_data,
        spatial_sum_multiplier_.gpu_data(), 0.,
        num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
        num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
        mean_.mutable_gpu_data());
  }

  // subtract mean
  channel_sub_kernel<Dtype> << <CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS >> >(
	  top[0]->count(), channels_, spatial_dim, top_data, mean_.gpu_data(), top_data);

  if (!use_global_stats_) {
    // compute variance using var(X) = E((X-EX)^2)
    caffe_gpu_mul(top[0]->count(), top_data, top_data, x_norm_.mutable_gpu_data());  // (X-EX)^2
    caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
        1. / (num * spatial_dim), x_norm_.gpu_data(),
        spatial_sum_multiplier_.gpu_data(), 0.,
        num_by_chans_.mutable_gpu_data());
    caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
        num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
        variance_.mutable_gpu_data());  // E((X_EX)^2)

    // compute and save moving average
    this->blobs_[2]->mutable_cpu_data()[0] *= moving_average_fraction_;
    this->blobs_[2]->mutable_cpu_data()[0] += 1;
    caffe_gpu_axpby(mean_.count(), Dtype(1), mean_.gpu_data(),
        moving_average_fraction_, this->blobs_[0]->mutable_gpu_data());
    int m = bottom[0]->count()/channels_;
    Dtype bias_correction_factor = m > 1 ? Dtype(m)/(m-1) : 1;
    caffe_gpu_axpby(variance_.count(), bias_correction_factor,
        variance_.gpu_data(), moving_average_fraction_,
        this->blobs_[1]->mutable_gpu_data());
  }

  // normalize variance
  caffe_gpu_add_scalar(variance_.count(), eps_, variance_.mutable_gpu_data());
  caffe_gpu_sqrt(variance_.count(), variance_.gpu_data(), variance_.mutable_gpu_data());

  channel_div_kernel_neps<Dtype> << <CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS >> >(
	  top[0]->count(), channels_, spatial_dim, top_data, variance_.gpu_data(), top_data);

  // TODO(cdoersch): The caching is only needed because later in-place layers
  //                 might clobber the data.  Can we skip this if they won't?
  caffe_copy(x_norm_.count(), top_data,
      x_norm_.mutable_gpu_data());
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff;
  if (bottom[0] != top[0]) {
    top_diff = top[0]->gpu_diff();
  } else {
    caffe_copy(x_norm_.count(), top[0]->gpu_diff(), x_norm_.mutable_gpu_diff());
    top_diff = x_norm_.gpu_diff();
  }
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  int num = bottom[0]->shape()[0];
  int spatial_dim = bottom[0]->count() / (channels_*bottom[0]->shape(0));
  if (use_global_stats_) {
    channel_div_kernel_neps<Dtype> << <CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS >> >(
		  top[0]->count(), channels_, spatial_dim, top_diff, variance_.gpu_data(), bottom_diff);
	  return;
  }
  const Dtype* top_data = x_norm_.gpu_data();
  // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
  //
  // dE(Y)/dX =
  //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
  //     ./ sqrt(var(X) + eps)
  //
  // where \cdot and ./ are hadamard product and elementwise division,
  // respectively, dE/dY is the top diff, and mean/var/sum are all computed
  // along all dimensions except the channels dimension.  In the above
  // equation, the operations allow for expansion (i.e. broadcast) along all
  // dimensions except the channels dimension where required.

  // sum(dE/dY \cdot Y)
  caffe_gpu_mul(top[0]->count(), top_data, top_diff, bottom_diff);
  caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
      bottom_diff, spatial_sum_multiplier_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
      num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
      mean_.mutable_gpu_data());

  // reshape (broadcast) the above
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), mean_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 0., bottom_diff);

  // sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_mul(top[0]->count(), top_data, bottom_diff, bottom_diff);

  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
      top_diff, spatial_sum_multiplier_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
      num_by_chans_.gpu_data(), batch_sum_multiplier_.gpu_data(), 0.,
      mean_.mutable_gpu_data());
  // reshape (broadcast) the above to make
  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.gpu_data(), mean_.gpu_data(), 0.,
      num_by_chans_.mutable_gpu_data());
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * channels_,
      spatial_dim, 1, 1., num_by_chans_.gpu_data(),
      spatial_sum_multiplier_.gpu_data(), 1., bottom_diff);

  // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
  caffe_gpu_axpby(top[0]->count(), Dtype(1), top_diff,
      Dtype(-1. / (num * spatial_dim)), bottom_diff);

  // note: temp_ still contains sqrt(var(X)+eps), computed during the forward
  // pass.
  channel_div_kernel_neps<Dtype> << <CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS >> >(
    top[0]->count(), channels_, spatial_dim, bottom_diff, variance_.gpu_data(), bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(BatchNormLayer);


}  // namespace caffe
