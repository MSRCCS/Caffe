#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/indexed_threshold_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class IndexedThresholdLossLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;

protected:
    IndexedThresholdLossLayerTest() :
        blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 3)),
        blob_bottom_index_(new Blob<Dtype>({ 10, 1 })),
        blob_top_loss_(new Blob<Dtype>()) {
        // fill the values
        FillerParameter filler_param;
        filler_param.set_min(0);
        filler_param.set_max(1);
        UniformFiller<Dtype> filler(filler_param);
        filler.Fill(this->blob_bottom_data_);
        blob_bottom_vec_.push_back(blob_bottom_data_);
        auto count = blob_bottom_data_->count(1);
        for (int i = 0; i < blob_bottom_index_->count(); ++i) {
            blob_bottom_index_->mutable_cpu_data()[i] = caffe_rng_rand() % count;
        }
        blob_bottom_vec_.push_back(blob_bottom_index_);
        blob_top_vec_.push_back(blob_top_loss_);
    }
    virtual ~IndexedThresholdLossLayerTest() {
        delete blob_bottom_data_;
        delete blob_bottom_index_;
        delete blob_top_loss_;
    }

    Blob<Dtype>* const blob_bottom_data_;
    Blob<Dtype>* const blob_bottom_index_;
    Blob<Dtype>* const blob_top_loss_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(IndexedThresholdLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(IndexedThresholdLossLayerTest, TestForward) {
    typedef typename TypeParam::Dtype Dtype;
    const Dtype kLossWeight = 3.7;
    const Dtype kNullScale = 0.5;
    const Dtype kPositiveScale = 5.0;
    const Dtype kThreshold = 0.3;

    LayerParameter layer_param;
    layer_param.add_loss_weight(kLossWeight);
    layer_param.mutable_indexedthreshold_loss_param()->set_null_scale(kNullScale);
    layer_param.mutable_indexedthreshold_loss_param()->set_positive_scale(kPositiveScale);
    layer_param.mutable_indexedthreshold_loss_param()->set_threshold(kThreshold);
    IndexedThresholdLossLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight = layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    Dtype loss = 0;
    int count = 0;
    auto channels = this->blob_bottom_data_->channels();
    auto height = this->blob_bottom_data_->height();
    auto width = this->blob_bottom_data_->width();
    auto outer_num = this->blob_bottom_data_->num();
    for (int i = 0; i < outer_num; ++i) {
        auto index = static_cast<int>(this->blob_bottom_index_->data_at({ i, 0 }));
        EXPECT_LT(index, channels * height * width);

        for (int j = 0; j < channels; ++j) {
            for (int k = 0; k < height; ++k) {
                for (int l = 0; l < width; ++l) {
                    auto prob = this->blob_bottom_data_->data_at(i, j, k, l);
                    if (j * height * width + k * width + l == index) {
                        if (prob < kThreshold) {
                            prob -= kThreshold;
                            loss += kPositiveScale * prob * prob;
                        }
                    } else {
                        loss += kNullScale * prob * prob;
                    }
                }
            }
        }
    }
    loss = loss / outer_num / Dtype(2);
    EXPECT_NEAR(loss, this->blob_top_loss_->cpu_data()[0], 1e-3);
    // check that loss is scaled appropriately by the objective weight.
    EXPECT_NEAR(loss * kLossWeight, loss_weight, 1e-5);
}

TYPED_TEST(IndexedThresholdLossLayerTest, TestGradient) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    IndexedThresholdLossLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_, 0);
}

}  // namespace caffe
