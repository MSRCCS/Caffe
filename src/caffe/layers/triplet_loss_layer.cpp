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
    triplets_.clear();
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

    map<uint64_t, Dtype> map_dist;

    Dtype loss = 0;
    for (int i = 0; i < triplets_.size(); i++)
    {
        Triplet<Dtype> &tri = triplets_[i];
        const Dtype *x_a = bottom_data + tri.a * dim;
        const Dtype *x_p = bottom_data + tri.p * dim;
        const Dtype *x_n = bottom_data + tri.n * dim;
        Dtype dis_ap, dis_an;
        uint64_t pair_ap = ((uint64_t)tri.a) << 32 | (uint32_t)tri.p;
        uint64_t pair_an = ((uint64_t)tri.a) << 32 | (uint32_t)tri.n;
        if (map_dist.find(pair_ap) != map_dist.end())
            dis_ap = map_dist[pair_ap];
        else
        {
            caffe_sub(dim, x_a, x_p, temp_sub);
            dis_ap = caffe_cpu_dot(dim, temp_sub, temp_sub);
            map_dist[pair_ap] = dis_ap;
        }
        if (map_dist.find(pair_an) != map_dist.end())
            dis_an = map_dist[pair_an];
        else
        {
            caffe_sub(dim, x_a, x_n, temp_sub);
            dis_an = caffe_cpu_dot(dim, temp_sub, temp_sub);
            map_dist[pair_an] = dis_an;
        }

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

      map<uint64_t, vector<int> > m_np, m_pa, m_an;
      // group n & p to get list a
      // group p & a to get list n
      // group a & n to get list p
      for (int i = 0; i < triplets_.size(); i++)
      {
          Triplet<Dtype> &tri = triplets_[i];
          if (tri.loss == 0)
              continue;
          uint64_t pair;
          pair = ((uint64_t)tri.n) << 32 | (uint32_t)tri.p;
          m_np[pair].push_back(tri.a);

          pair = ((uint64_t)tri.p) << 32 | (uint32_t)tri.a;
          m_pa[pair].push_back(tri.n);

          pair = ((uint64_t)tri.a) << 32 | (uint32_t)tri.n;
          m_an[pair].push_back(tri.p);
      }

      map<uint64_t, vector<int> >::iterator iter;

      // dx_a += x_n - x_p
      for (iter = m_np.begin(); iter != m_np.end(); iter++)
      {
          uint64_t pair = iter->first;
          vector<int>& list = iter->second;
          int _n = pair >> 32;
          int _p = pair & 0xffffffff;
          caffe_sub(dim, bottom_data + _n * dim, bottom_data + _p * dim, temp_sub);
          for (int i = 0; i < list.size(); i++)
              caffe_axpy(dim, (Dtype)1, temp_sub, bottom_diff + list[i] * dim);
      }

      // dx_p += x_p - x_a
      for (iter = m_pa.begin(); iter != m_pa.end(); iter++)
      {
          uint64_t pair = iter->first;
          vector<int>& list = iter->second;
          int _p = pair >> 32;
          int _a = pair & 0xffffffff;
          caffe_sub(dim, bottom_data + _p * dim, bottom_data + _a * dim, temp_sub);
          caffe_axpy(dim, (Dtype)list.size(), temp_sub, bottom_diff + _p * dim);
      }

      // dx_n += x_a - x_n
      for (iter = m_an.begin(); iter != m_an.end(); iter++)
      {
          uint64_t pair = iter->first;
          vector<int>& list = iter->second;
          int _a = pair >> 32;
          int _n = pair & 0xffffffff;
          caffe_sub(dim, bottom_data + _a * dim, bottom_data + _n * dim, temp_sub);
          caffe_axpy(dim, (Dtype)list.size(), temp_sub, bottom_diff + _n * dim);
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
