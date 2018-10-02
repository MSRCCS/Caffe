#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#include "caffe/layers/yolo_bbs_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class YoloBBsLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
protected:
  YoloBBsLayerTest() :
      blob_xy_(new Blob<Dtype>(2, 3 * 2, 7, 7)),
      blob_wh_(new Blob<Dtype>(2, 3 * 2, 7, 7)),
      blob_imageinfo_(new Blob<Dtype>(1, 2, 1, 1)),
      blob_top_bbs_(new Blob<Dtype>()) {
      //// fill the values
      FillerParameter filler_param;
      filler_param.set_min(0);
      filler_param.set_max(1);
      UniformFiller<Dtype> filler(filler_param);
      filler.Fill(blob_xy_);
      filler.Fill(blob_wh_);

      SigmoidForward(*blob_xy_);

      blob_bottom_vec_.push_back(blob_xy_);
      blob_bottom_vec_.push_back(blob_wh_);
      blob_bottom_vec_.push_back(blob_imageinfo_);
      blob_top_vec_.push_back(blob_top_bbs_);
  }

  void SigmoidForward(Blob<Dtype> &blob) {
      LayerParameter param_sig_xy;
      shared_ptr<SigmoidLayer<Dtype>> sig_layer_xy(
          new SigmoidLayer<Dtype>(param_sig_xy));
      std::vector<Blob<Dtype>*> sig_layer_xy_bottom;
      std::vector<Blob<Dtype>*> sig_layer_xy_top;
      sig_layer_xy_bottom.push_back(&blob);
      sig_layer_xy_top.push_back(&blob);
      sig_layer_xy->LayerSetUp(sig_layer_xy_bottom, sig_layer_xy_top);
      sig_layer_xy->Reshape(sig_layer_xy_bottom, sig_layer_xy_top);
      sig_layer_xy->Forward(sig_layer_xy_bottom, sig_layer_xy_top);
  }

  virtual ~YoloBBsLayerTest() {
      delete blob_xy_;
      delete blob_wh_;
      delete blob_imageinfo_;
      delete blob_top_bbs_;
  }
  Blob<Dtype>* const blob_xy_;
  Blob<Dtype>* const blob_wh_;
  Blob<Dtype>* const blob_imageinfo_;
  Blob<Dtype>* const blob_top_bbs_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(YoloBBsLayerTest, TestDtypesAndDevices);

TYPED_TEST(YoloBBsLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;

  const int kImageWidth = 315;
  const int kImageHeight = 416;
  this->blob_imageinfo_->mutable_cpu_data()[0] = kImageHeight;
  this->blob_imageinfo_->mutable_cpu_data()[1] = kImageWidth;

  vector<float> biases = { 0.77871f, 1.14074f, 3.00525f, 4.31277f, 9.22725f, 9.61974f };
  LayerParameter layer_param;
  auto yolobbs_param = layer_param.mutable_yolobbs_param();
  for (int i = 0; i < biases.size(); i++) {
      yolobbs_param->add_biases(biases[i]);
  }
  const int kFeatStride = 32;
  yolobbs_param->set_feat_stride(kFeatStride);

  scoped_ptr<YoloBBsLayer<Dtype> > layer(new YoloBBsLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

  int batches = this->blob_xy_->num();
  int height = this->blob_xy_->height();
  int width = this->blob_xy_->width();
  int num_anchor = this->blob_xy_->channels() / 2;
  int net_w = kFeatStride * width;
  int net_h = kFeatStride * height;

  auto bbs_data = this->blob_top_bbs_->cpu_data();
  vector<Dtype> bbox(4);
  auto& x = bbox[0];
  auto& y = bbox[1];
  auto& w = bbox[2];
  auto& h = bbox[3];

  EXPECT_GT((Dtype)net_w / kImageWidth, (Dtype)net_h / kImageHeight);
  int new_h = net_h;
  int new_w = (kImageWidth * net_h) / kImageHeight;

  for (int b = 0; b < batches; b++) {
      for (int n = 0; n < num_anchor; n++) {
          for (int j = 0; j < height; j++) {
              for (int i = 0; i < width; i++) {
                  x = (this->blob_xy_->data_at(b, n, j, i) + i) / width;
                  y = (this->blob_xy_->data_at(b, n + num_anchor, j, i) + j) / height;
                  w = exp(this->blob_wh_->data_at(b, n, j, i)) * biases[2 * n] / width;
                  h = exp(this->blob_wh_->data_at(b, n + num_anchor, j, i)) * biases[2 * n + 1] / height;

                  // Correct the box
                  x = (x - (net_w - new_w) / 2. / net_w) / ((Dtype)new_w / net_w);
                  y = (y - (net_h - new_h) / 2. / net_h) / ((Dtype)new_h / net_h);
                  w *= (Dtype)net_w / new_w;
                  h *= (Dtype)net_h / new_h;
                  x *= kImageWidth;
                  w *= kImageWidth;
                  y *= kImageHeight;
                  h *= kImageHeight;

                  for (int k = 0; k < 4; ++k)
                    EXPECT_FLOAT_EQ(bbox[k], *(bbs_data + this->blob_top_bbs_->offset({ b, n, j, i, k })));
              }
          }
      }
  }
}

}
