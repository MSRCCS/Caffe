#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/l2_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void L2NormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / num;

  const Dtype* x = bottom[0]->gpu_data();
  const Dtype* sum_mult_data = sum_multiplier_.gpu_data();

  // use top_diff as temporal space to store squred data of bottom_data.
  // note that top_data may be the same as bottom_data if the computation is inplace.
  Dtype* sqr_data = top[0]->mutable_gpu_diff();
  caffe_gpu_powx(num * dim, x, Dtype(2), sqr_data);
  caffe_gpu_gemv(CblasNoTrans, num, dim, Dtype(1), sqr_data, sum_mult_data, Dtype(0), norm_.mutable_gpu_data());

  const Dtype* norm_data = norm_.cpu_data();
  Dtype* y = top[0]->mutable_gpu_data();
  for (int i = 0; i < num; ++i)
    caffe_gpu_scale<Dtype>(dim, pow(norm_data[i], -0.5), x + i * dim, y + i * dim);
}
  
template <typename Dtype>
void L2NormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int num = top[0]->num();
  int dim = top[0]->count() / num;

  const Dtype* dy = top[0]->gpu_diff();
  const Dtype* y = top[0]->gpu_data();
  const Dtype* norm_data = norm_.cpu_data();
  Dtype* dx = bottom[0]->mutable_gpu_diff();

  // scale y inplace as we won't need y in back propagation
  Dtype* scaled_y = top[0]->mutable_gpu_data();
  for (int i = 0; i < num; ++i)
  {
    Dtype scale;
    caffe_gpu_dot(dim, dy, y, &scale);
    caffe_gpu_scale(dim, scale, y, scaled_y);
    caffe_gpu_sub(dim, dy, scaled_y, dx);
    caffe_gpu_scale(dim, Dtype(pow(norm_data[i], -0.5)), dx, dx);

    y += dim;
    dy += dim;
    dx += dim;
    scaled_y += dim;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(L2NormLayer);
}  // namespace caffe
