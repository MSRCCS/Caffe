#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
//#include "caffe/vision_layers.hpp"
#include "caffe/layers/xcov_loss_layer.hpp" 
namespace caffe {


template <typename Dtype>
void XCovLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  mean_vec_.clear();
  mean_vec_.push_back(&mean_0_);
  mean_vec_.push_back(&mean_1_);

  temp_vec_.clear();
  temp_vec_.push_back(&temp_0_);
  temp_vec_.push_back(&temp_1_);
}


template <typename Dtype>
void XCovLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->num();
  top[0]->Reshape(1, 1, 1, 1);

  int dim0 = bottom[0]->count() / num;
  int dim1 = bottom[1]->count() / num;
  xcov_.Reshape(dim0*dim1, 1, 1, 1);
  if (this->layer_param_.xcov_param().report_covmat()) {
    CHECK_EQ(top.size(), 2) << "You must specify two top blobs to output the diagonal sum.";
    top[1]->Reshape(1, dim0, dim1, 1);
  }

  for (int i = 0 ; i < bottom.size() ; i++) {
    mean_vec_[i]->Reshape(1, bottom[i]->channels(),
        bottom[i]->height(), bottom[i]->width());
    temp_vec_[i]->Reshape(bottom[i]->num(), bottom[i]->channels(),
        bottom[i]->height(), bottom[i]->width());
  }
  batch_sum_multiplier_.Reshape(bottom[0]->num(), 1, 1, 1);
  Dtype* batch_multiplier_data = batch_sum_multiplier_.mutable_cpu_data();
  caffe_set(batch_sum_multiplier_.count(), Dtype(1), batch_multiplier_data);
}

template <typename Dtype>
void XCovLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // for now, we support only two inputs
  CHECK_EQ(bottom.size(), 2);

  for (int i = 0 ; i < bottom.size() ; i++) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    int num = bottom[i]->num();
    int dim = bottom[i]->count() / num;

    // calculate mean vector over batch
    caffe_cpu_gemv<Dtype>(CblasTrans, num, dim, 1. / num, bottom_data,
        batch_sum_multiplier_.cpu_data(), 0., mean_vec_[i]->mutable_cpu_data());

    // broadcast and negative mean vector
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
        batch_sum_multiplier_.cpu_data(),
        mean_vec_[i]->cpu_data(),
        0.,
        temp_vec_[i]->mutable_cpu_data());

    // subtract mean
    caffe_add(temp_vec_[i]->count(), bottom_data, temp_vec_[i]->cpu_data(),
        temp_vec_[i]->mutable_cpu_data());
  }

  int num = bottom[0]->num();
  int dim0 = bottom[0]->count() / num;
  int dim1 = bottom[1]->count() / num;

  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, dim0, dim1, num, 1./num,
      temp_vec_[0]->cpu_data(),
      temp_vec_[1]->cpu_data(),
      0.,
      xcov_.mutable_cpu_data());

  // report the covariance matrix before any modifications are made
  if (this->layer_param_.xcov_param().report_covmat()) {
    caffe_copy(xcov_.count(), xcov_.cpu_data(), top[1]->mutable_cpu_data());
  }

  // Filter a band of the matrix around the diagonal by filling xcov_.data in
  // from the covariance matrix cached in xcov_.diff.
  // Note that these operations do copy the diagonal, so it may still need to be
  // subtracted.
  int block_width = this->layer_param_.xcov_param().block_width();
  int band_width = this->layer_param_.xcov_param().band_width();
  int mindim = std::min(dim0, dim1);
  // TODO: filter using the GPU, not CPU
  if (band_width != -1) {
    caffe_copy(xcov_.count(), xcov_.cpu_data(), xcov_.mutable_cpu_diff());
    caffe_set(xcov_.count(), Dtype(0), xcov_.mutable_cpu_data());
    for (int row = 0; row < mindim; row++) {
      int col_start = std::max(row-band_width, 0);
      int col_end = std::min(row+band_width+1, dim0);
      for (int col = col_start; col < col_end; col++) {
        xcov_.mutable_cpu_data()[row*dim0 + col] = xcov_.cpu_diff()[row*dim0 + col];
      }
    }
  }

  // these set certain elements to 0 instead of assuming everything is 0 then filling in
  if (block_width != -1) {
    for (int block_offset = 0;
         block_width * block_offset < mindim;
         block_offset++) {
      for (int row = block_offset * block_width;
           row < std::min(mindim, (block_offset+1) * block_width);
           row++) {
        for (int col = block_offset * block_width;
             col < std::min(mindim, (block_offset+1) * block_width);
             col++) {
          xcov_.mutable_cpu_data()[row*dim0 + col] = Dtype(0);
        }
      }
    }
  }
  // subtract diagonal TODO: this is redundant
  if (this->layer_param_.xcov_param().subtract_diagonal()) {
    for (int i = 0; i < mindim; i++) {
      xcov_.mutable_cpu_data()[i*dim0 + i] = Dtype(0);
    }
  }

  // square terms in xcov
  Dtype dot = caffe_cpu_dot<Dtype>(xcov_.count(), xcov_.cpu_data(),
      xcov_.cpu_data());

  Dtype loss = dot / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void XCovLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  for (int i = 0 ; i < bottom.size() ; i++) {
    int num = bottom[i]->num();
    int dim = bottom[i]->count() / num;

    // calculate mean of (z^m_j - \bar{z_j}) over batch (over m)
    caffe_cpu_gemv<Dtype>(CblasTrans, num, dim, 1. / num, temp_vec_[i]->cpu_data(),
        batch_sum_multiplier_.cpu_data(), 0., mean_vec_[i]->mutable_cpu_data());

    // broadcast this new mean, then subtract from each temp
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
        batch_sum_multiplier_.cpu_data(),
        mean_vec_[i]->cpu_data(),
        1.,
        temp_vec_[i]->mutable_cpu_data());
  }

  const Dtype top_diff = top[0]->cpu_diff()[0];

  Dtype* bottom_diff_0 = bottom[0]->mutable_cpu_diff();
  Dtype* bottom_diff_1 = bottom[1]->mutable_cpu_diff();

  int num = bottom[0]->num();
  int dim0 = bottom[0]->count() / num;
  int dim1 = bottom[1]->count() / num;

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num, dim0, dim1,
      top_diff/num,
      temp_vec_[1]->cpu_data(),
      xcov_.cpu_data(),
      0.,
      bottom_diff_0);

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim1, dim0,
      top_diff/num,
      temp_vec_[0]->cpu_data(),
      xcov_.cpu_data(),
      0.,
      bottom_diff_1);
}


#ifdef CPU_ONLY
STUB_GPU(XCovLossLayer);
#endif

INSTANTIATE_CLASS(XCovLossLayer);
REGISTER_LAYER_CLASS(XCovLoss);

}  // namespace caffe
