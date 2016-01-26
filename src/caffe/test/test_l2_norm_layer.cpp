#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/l2_norm_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class L2NormLayerTest : public MultiDeviceTest<TypeParam>
{
  typedef typename TypeParam::Dtype Dtype;
protected:
  L2NormLayerTest()
      : blob_bottom_(new Blob<Dtype>(4, 5, 1, 1)),
        blob_top_(new Blob<Dtype>())
  {
    Caffe::set_random_seed(1701);
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~L2NormLayerTest()
  { 
    delete blob_bottom_; 
    delete blob_top_; 
  }

  void TestForward()
  {
    typedef typename TypeParam::Dtype Dtype;
    double precision = 1e-5;
    LayerParameter layer_param;
    L2NormLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

    int num = this->blob_bottom_->num();
    int dim = this->blob_bottom_->count() / num;

    Blob<Dtype> bottom;
    bottom.ReshapeLike(*blob_bottom_);
    caffe_copy(num * dim, this->blob_bottom_->cpu_data(), bottom.mutable_cpu_data());

    vector<Dtype> normsqr_x(num);
    for (int i = 0; i < num; ++i)
    {
      const Dtype* x = this->blob_bottom_->cpu_data() + i * dim;
      normsqr_x[i] = 0;
      for (int j = 0; j < dim; ++j)
        normsqr_x[i] += x[j] * x[j];
    }

    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    for (int i = 0; i < num; ++i)
    {
      const Dtype* x = bottom.cpu_data() + i * dim;
      const Dtype* y = this->blob_top_->cpu_data() + i * dim;

      Dtype normsqr_y = 0;
      for (int j = 0; j < dim; ++j)
        normsqr_y += y[j] * y[j];

      EXPECT_NEAR(normsqr_y, 1, precision);

      Dtype c = pow(normsqr_x[i], -0.5);
      for (int j = 0; j < dim; ++j)
        EXPECT_NEAR(y[j], x[j] * c, precision);
    }
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(L2NormLayerTest, TestDtypesAndDevices);

TYPED_TEST(L2NormLayerTest, TestForward)
{
  this->TestForward();
}

TYPED_TEST(L2NormLayerTest, TestGradient)
{
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  L2NormLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(L2NormLayerTest, TestL2NormInPlace) {
  typedef typename TypeParam::Dtype Dtype;
  // Set layer parameters
  LayerParameter l2_norm_layer_param;
  L2NormLayer<Dtype> l2norm(l2_norm_layer_param);
  L2NormLayer<Dtype> l2norm2(l2_norm_layer_param);
  // Set up blobs
  vector<Blob<Dtype>*> blob_bottom_vec_2;
  vector<Blob<Dtype>*> blob_top_vec_2;
  shared_ptr<Blob<Dtype> > blob_bottom_2(new Blob<Dtype>());
  shared_ptr<Blob<Dtype> > blob_top_2(new Blob<Dtype>());
  blob_bottom_vec_2.push_back(blob_bottom_2.get());
  blob_top_vec_2.push_back(blob_top_2.get());
  blob_bottom_2->CopyFrom(*this->blob_bottom_, false, true);
  // SetUp layers
  l2norm.SetUp(this->blob_bottom_vec_, this->blob_bottom_vec_);   // inplace
  l2norm2.SetUp(blob_bottom_vec_2, blob_top_vec_2);

  // Forward non-in-place
  l2norm2.Forward(blob_bottom_vec_2, blob_top_vec_2);
  // Forward in-place
  l2norm.Forward(this->blob_bottom_vec_, this->blob_bottom_vec_);

  // Check numbers
  for (int s = 0; s < blob_top_2->count(); ++s) {
    EXPECT_EQ(this->blob_bottom_->cpu_data()[s], blob_top_2->cpu_data()[s]);
  }
  // Fill top diff with random numbers
  shared_ptr<Blob<Dtype> > tmp_blob(new Blob<Dtype>());
  tmp_blob->ReshapeLike(*blob_top_2.get());
  FillerParameter filler_param;
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(tmp_blob.get());
  caffe_copy(blob_top_2->count(), tmp_blob->cpu_data(), 
      this->blob_bottom_->mutable_cpu_diff());
  caffe_copy(blob_top_2->count(), tmp_blob->cpu_data(),
      blob_top_2->mutable_cpu_diff());

  // Backward non-in-place
  vector<bool> propagate_down;
  propagate_down.push_back(true);
  l2norm2.Backward(blob_top_vec_2, propagate_down, blob_bottom_vec_2);
  // Backward in-place
  l2norm.Backward(this->blob_bottom_vec_, propagate_down, this->blob_bottom_vec_);

  // Check numbers
  for (int s = 0; s < blob_bottom_2->count(); ++s) {
    EXPECT_EQ(this->blob_bottom_->cpu_diff()[s], blob_bottom_2->cpu_diff()[s]);
  }
}

}  // namespace caffe