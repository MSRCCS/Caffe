#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/softmaxtree_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SoftmaxTreeLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SoftmaxTreeLayerTest()
      : tree_file_name_(CMAKE_SOURCE_DIR "caffe/test/test_data/11n_4g.tree"), 
        group_size_{4, 2, 3, 2},
        blob_bottom_(new Blob<Dtype>(2, 11, 2, 3)),
        blob_top_(new Blob<Dtype>()) {
      blob_bottom_vec_.push_back(blob_bottom_);
      blob_top_vec_.push_back(blob_top_);
      FillBottom();
  }
  void FillBottom() {
      // fill the values
      FillerParameter filler_param;
      GaussianFiller<Dtype> filler(filler_param);
      filler.Fill(blob_bottom_);
  }
  void Initialize(vector<int> group_size, const char* tree_file_name) {
      ASSERT_STRNE(tree_file_name, NULL);
      tree_file_name_ = tree_file_name;
      group_size_ = group_size;
      int n = 0;
      for (auto g : group_size_)
          n += g;
      blob_bottom_->Reshape(2, n, 2, 3);
      FillBottom();
  }
  void TestForward() {
      // Test sum
      for (int i = 0; i < blob_bottom_->num(); ++i) {
          for (int k = 0; k < blob_bottom_->height(); ++k) {
              for (int l = 0; l < blob_bottom_->width(); ++l) {
                  int offset = 0;
                  for (int g = 0; g < group_size_.size(); ++g) {
                      int size = group_size_[g];
                      Dtype sum = 0;
                      for (int j = 0; j < size; ++j) {
                          sum += blob_top_->data_at(i, offset + j, k, l);
                      }
                      EXPECT_NEAR(sum, 1.0, 1e-3);
                      // Test exact values
                      Dtype scale = 0;
                      for (int j = 0; j < size; ++j) {
                          scale += exp(blob_bottom_->data_at(i, offset + j, k, l));
                      }
                      for (int j = 0; j < size; ++j) {
                          EXPECT_NEAR(blob_top_->data_at(i, offset + j, k, l),
                                      exp(blob_bottom_->data_at(i, offset + j, k, l)) / scale, 1e-3)
                              << "debug: " << i << " " << offset + j;
                      }

                      offset += size;
                  }
              }
          }
      }
  }
  virtual ~SoftmaxTreeLayerTest() { delete blob_bottom_; delete blob_top_; }
  string tree_file_name_;
  vector<int> group_size_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SoftmaxTreeLayerTest, TestDtypesAndDevices);

TYPED_TEST(SoftmaxTreeLayerTest, TestForward) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_param()->set_tree(this->tree_file_name_);
    SoftmaxTreeLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    this->TestForward();
}

TYPED_TEST(SoftmaxTreeLayerTest, TestGradient) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_param()->set_tree(this->tree_file_name_);
    SoftmaxTreeLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                    this->blob_top_vec_);
}

TYPED_TEST(SoftmaxTreeLayerTest, TestForwardSubGroups) {
    typedef typename TypeParam::Dtype Dtype;
    this->Initialize({ 2, 2, 2, 3, 2, 2, 2 }, CMAKE_SOURCE_DIR "caffe/test/test_data/15n_5g_7s.tree");
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_param()->set_tree(this->tree_file_name_);
    SoftmaxTreeLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    this->TestForward();
}

TYPED_TEST(SoftmaxTreeLayerTest, TestGradientSubGroups) {
    typedef typename TypeParam::Dtype Dtype;
    this->Initialize({ 2, 2, 2, 3, 2, 2, 2 }, CMAKE_SOURCE_DIR "caffe/test/test_data/15n_5g_7s.tree");
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_param()->set_tree(this->tree_file_name_);
    SoftmaxTreeLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
                                    this->blob_top_vec_);
}

}  // namespace caffe
