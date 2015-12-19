#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

//#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/multi_label_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

//using boost::scoped_ptr;

namespace caffe {

template <typename Dtype>
class MultiLabelLossLayerTest : public CPUDeviceTest<Dtype> {
 protected:
  MultiLabelLossLayerTest()
	  : blob_bottom_data_(new Blob<Dtype>(10, 5 * 2, 1, 1)),
		blob_bottom_label_(new Blob<Dtype>(10, 3, 1, 1)),
		blob_top_loss_(new Blob<Dtype>()) {
	// fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 5;
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~MultiLabelLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MultiLabelLossLayerTest, TestDtypes);

TYPED_TEST(MultiLabelLossLayerTest, TestGradient) {
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  MultiLabelLossLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(MultiLabelLossLayerTest, TestGradientUnnormalized) {
  LayerParameter layer_param;
  layer_param.mutable_loss_param()->set_normalize(false);
  MultiLabelLossLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
