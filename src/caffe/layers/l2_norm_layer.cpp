#include <algorithm>
#include <vector>
#include <cmath>

#include "caffe/layers/l2_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void L2NormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);

  vector<int> mult_dims(1, bottom[0]->count(1));
  sum_multiplier_.Reshape(mult_dims);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);

  vector<int> norm_dims(1, bottom[0]->num());
  norm_.Reshape(norm_dims);
}

template <typename Dtype>
void L2NormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / num;

  const Dtype* x = bottom[0]->cpu_data();
  const Dtype* sum_mult_data = sum_multiplier_.cpu_data();
  Dtype* norm_data = norm_.mutable_cpu_data();

  // use top_diff as temporal space to store squred data of bottom_data.
  // note that top_data may be the same as bottom_data if the computation is inplace.
  Dtype* sqr_data = top[0]->mutable_cpu_diff();
  caffe_sqr<Dtype>(num * dim, x, sqr_data);
  caffe_cpu_gemv(CblasNoTrans, num, dim, Dtype(1), sqr_data, sum_mult_data, Dtype(0), norm_data);

  Dtype* y = top[0]->mutable_cpu_data();
  for (int i = 0; i < num; ++i)
    caffe_cpu_scale<Dtype>(dim, pow(norm_data[i], -0.5), x + i * dim, y + i * dim);
}

template <typename Dtype>
void L2NormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int num = top[0]->num();
  int dim = top[0]->count() / num;

  const Dtype* dy = top[0]->cpu_diff();
  const Dtype* y = top[0]->cpu_data();
  const Dtype* norm_data = norm_.cpu_data();
  Dtype* dx = bottom[0]->mutable_cpu_diff();

  // scale y inplace as we won't need y in back propagation
  Dtype* scaled_y = top[0]->mutable_cpu_data();
  for (int i = 0; i < num; ++i)
  {
    Dtype scale = caffe_cpu_dot(dim, dy, y);
    caffe_cpu_scale(dim, scale, y, scaled_y);
    caffe_sub(dim, dy, scaled_y, dx);
    caffe_cpu_scale(dim, Dtype(pow(norm_data[i], -0.5)), dx, dx);

    y += dim;
    dy += dim;
    dx += dim;
    scaled_y += dim;
  }
}

#ifdef CPU_ONLY
STUB_GPU(L2NormLayer);
#endif

INSTANTIATE_CLASS(L2NormLayer);
REGISTER_LAYER_CLASS(L2Norm);

}  // namespace caffe
