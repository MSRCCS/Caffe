#include <cmath>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/wsgm_loss_layer.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class WSgmLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

protected:
  WSgmLossLayerTest()
    : blob_bottom_data_(new Blob<Dtype>(5, 5, 1, 1)),
    blob_bottom_label_(new Blob<Dtype>(5, 1, 1, 1)),
    blob_bottom_prob_(new Blob<Dtype>(5, 5, 1, 1)),
    blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    filler.Fill(this->blob_bottom_prob_);
    int n_cls = 5;
    Dtype* prob_data = blob_bottom_prob_->mutable_cpu_data();
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      Dtype sum_pksqr = caffe_cpu_asum<Dtype>(n_cls, prob_data + blob_bottom_prob_->offset(i));
      sum_pksqr = Dtype(1.0) / sum_pksqr;
      caffe_scal(n_cls, sum_pksqr, prob_data + blob_bottom_prob_->offset(i));
    }
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    shared_ptr<Caffe::RNG> rng(new Caffe::RNG(prefetch_rng_seed));
    caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(rng->generator());
    Dtype* label_data = blob_bottom_label_->mutable_cpu_data();
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      label_data[i] = (*prefetch_rng)() % n_cls;
    }
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_bottom_vec_.push_back(blob_bottom_prob_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~WSgmLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_top_loss_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_bottom_prob_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(WSgmLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(WSgmLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  int n_cls = 5;
  WSgmParameter *wsgmloss_layer_param = layer_param.mutable_wsgm_param();
  wsgmloss_layer_param->set_num_classes(n_cls);
  layer_param.add_loss_weight(1);
  WSgmLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
    this->blob_top_vec_, 0);
}

}  // namespace caffe
