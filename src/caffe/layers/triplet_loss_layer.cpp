#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/triplet_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template struct Triplet<float>;
template struct Triplet<double>;

template <typename Dtype>
void TripletLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
}

template <typename Dtype>
void TripletLossLayer<Dtype>::CollectTripets(Blob<Dtype>& label_blob)
{
    int num = label_blob.num();
    const Dtype* label = label_blob.cpu_data();
    const int pos_label_value = static_cast<int>(label[0]); // assume the first one is positive label
    int pos_label_end = 0;
    while (pos_label_end < num)
    {
        int label_value = static_cast<int>(label[++pos_label_end]);
        if (label_value != pos_label_value)
            break;
    }
    int neg_label_start = pos_label_end;
    for (int a = 0; a < pos_label_end; a++)
    {
        for (int p = a + 1; p < pos_label_end; p++)
        {
            for (int n = neg_label_start; n < num; n++)
            {
                Triplet<Dtype> tri(a, p, n);
                triplets_.push_back(tri);
            }
        }
    }
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    CollectTripets(*bottom[1]);
    Dtype alpha = this->layer_param().triplet_loss_param().margin();

    int num = bottom[0]->num();
    int dim = bottom[0]->count() / num;
    const Dtype* bottom_data = bottom[0]->cpu_data();

    Blob<Dtype> temp_sub_blob(1, dim, 1, 1);
    Dtype* temp_sub = temp_sub_blob.mutable_cpu_data();

    Dtype loss = 0;
    for (int i = 0; i < triplets_.size(); i++)
    {
        Triplet<Dtype> &tri = triplets_[i];
        const Dtype *x_a = bottom_data + tri.a * dim;
        const Dtype *x_p = bottom_data + tri.p * dim;
        const Dtype *x_n = bottom_data + tri.n * dim;
        caffe_sub(dim, x_a, x_p, temp_sub);
        Dtype dis_ap = caffe_cpu_dot(dim, temp_sub, temp_sub);
        caffe_sub(dim, x_a, x_n, temp_sub);
        Dtype dis_an = caffe_cpu_dot(dim, temp_sub, temp_sub);
        tri.loss = std::max(Dtype(0), dis_ap - dis_an + alpha);
        loss += tri.loss;
    }

    top[0]->mutable_cpu_data()[0] = loss / triplets_.size();
}

template <typename Dtype>
void TripletLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
      int num = bottom[0]->num();
      int dim = bottom[0]->count() / num;
      const Dtype* bottom_data = bottom[0]->cpu_data();
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      caffe_set(num * dim, Dtype(0), bottom_diff);

      Blob<Dtype> temp_sub_blob(1, dim, 1, 1);
      Dtype* temp_sub = temp_sub_blob.mutable_cpu_data();

      for (int i = 0; i < triplets_.size(); i++)
      {
          Triplet<Dtype> &tri = triplets_[i];
          if (tri.loss == 0)
              continue;
          const Dtype *x_a = bottom_data + tri.a * dim;
          const Dtype *x_p = bottom_data + tri.p * dim;
          const Dtype *x_n = bottom_data + tri.n * dim;
          Dtype *dx_a = bottom_diff + tri.a * dim;
          Dtype *dx_p = bottom_diff + tri.p * dim;
          Dtype *dx_n = bottom_diff + tri.n * dim;

          caffe_sub(dim, x_n, x_p, temp_sub);
          caffe_add(dim, dx_a, temp_sub, dx_a);

          caffe_sub(dim, x_p, x_a, temp_sub);
          caffe_add(dim, dx_p, temp_sub, dx_p);

          caffe_sub(dim, x_a, x_n, temp_sub);
          caffe_add(dim, dx_n, temp_sub, dx_n);
      }
      // Scale gradient
      Dtype loss_weight = top[0]->cpu_diff()[0] * 2 / triplets_.size(); // 2 is from dL_dxa, dL_dxp, dL_dxn
      caffe_scal(num * dim, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(TripletLossLayer);
#endif

INSTANTIATE_CLASS(TripletLossLayer);
REGISTER_LAYER_CLASS(TripletLoss);

}  // namespace caffe
