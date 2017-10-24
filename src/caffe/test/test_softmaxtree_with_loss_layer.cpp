#include <cmath>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/softmaxtree_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class SoftmaxTreeWithLossLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;

protected:
    SoftmaxTreeWithLossLayerTest() :
        tree_file_name_(CMAKE_SOURCE_DIR "caffe/test/test_data/11n_4g.tree"),
        parent_{ -1, -1, -1, -1, 0, 0, 4, 4, 4, 8, 8 },
        blob_bottom_data_(new Blob<Dtype>(5, 11, 2, 3)),
        blob_bottom_label_(new Blob<Dtype>(5, 1, 2, 3)),
        blob_bottom_label_perbatch_(new Blob<Dtype>({ 5, 1 })),
        blob_bottom_objectness_(new Blob<Dtype>(5, 1, 2, 3)),
        blob_top_loss_(new Blob<Dtype>()),
        blob_top_prob_(new Blob<Dtype>(5, 11, 2, 3)) {
        // fill the values
        FillerParameter filler_param;
        filler_param.set_std(10);
        GaussianFiller<Dtype> filler(filler_param);
        filler.Fill(blob_bottom_data_);
        blob_bottom_vec_.push_back(blob_bottom_data_);
        for (int i = 0; i < blob_bottom_label_->count(); ++i) {
            blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 11;
        }
        for (int i = 0; i < blob_bottom_label_perbatch_->count(); ++i) {
            blob_bottom_label_perbatch_->mutable_cpu_data()[i] = caffe_rng_rand() % 11;
        }
        FillerParameter uni_filler_param;
        uni_filler_param.set_min(0);
        uni_filler_param.set_max(1);
        UniformFiller<Dtype> uni_filler(uni_filler_param);
        uni_filler.Fill(blob_bottom_objectness_);

        blob_bottom_vec_.push_back(blob_bottom_label_);
        blob_top_vec_.push_back(blob_top_loss_);
    }
    virtual ~SoftmaxTreeWithLossLayerTest() {
        delete blob_bottom_data_;
        delete blob_bottom_label_;
        delete blob_bottom_label_perbatch_;
        delete blob_bottom_objectness_;
        delete blob_top_loss_;
        delete blob_top_prob_;
    }
    string tree_file_name_;
    vector<int> parent_;
    Blob<Dtype>* const blob_bottom_data_;
    Blob<Dtype>* const blob_bottom_label_;
    Blob<Dtype>* const blob_bottom_label_perbatch_;
    Blob<Dtype>* const blob_bottom_objectness_;
    Blob<Dtype>* const blob_top_loss_;
    Blob<Dtype>* const blob_top_prob_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SoftmaxTreeWithLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestGradient) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_param()->set_tree(tree_file_name_);
    layer_param.add_loss_weight(3);
    SoftmaxTreeWithLossLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    checker.CheckGradientExhaustive(&layer, blob_bottom_vec_, blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestForward) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_param()->set_tree(tree_file_name_);
    layer_param.mutable_loss_param()->set_normalization(LossParameter_NormalizationMode_NONE);
    layer_param.add_loss_weight(1);
    layer_param.add_loss_weight(0);
    // Need shared probability blob at the top
    blob_top_vec_.push_back(blob_top_prob_);

    scoped_ptr<SoftmaxTreeWithLossLayer<Dtype> > layer(new SoftmaxTreeWithLossLayer<Dtype>(layer_param));
    layer->SetUp(blob_bottom_vec_, blob_top_vec_);
    layer->Forward(blob_bottom_vec_, blob_top_vec_);
    Dtype loss = 0;
    int count = 0;
    for (int i = 0; i < blob_bottom_data_->num(); ++i) {
        for (int k = 0; k < blob_bottom_data_->height(); ++k) {
            for (int l = 0; l < blob_bottom_data_->width(); ++l) {
                int label_value = static_cast<int>(blob_bottom_label_->data_at(i, 0, k, l));
                EXPECT_GE(label_value, 0);
                EXPECT_LT(label_value, blob_top_prob_->shape(1));
                while (label_value >= 0) {
                    loss -= log(std::max(blob_top_prob_->data_at(i, label_value, k, l), Dtype(FLT_MIN)));
                    ++count;
                    label_value = parent_[label_value];
                }
            }
        }
    }
    EXPECT_NEAR(loss, blob_top_loss_->cpu_data()[0], 1e-3);

    layer_param.mutable_loss_param()->set_normalization(LossParameter_NormalizationMode_VALID);
    layer.reset(new SoftmaxTreeWithLossLayer<Dtype>(layer_param));
    layer->SetUp(blob_bottom_vec_, blob_top_vec_);
    layer->Forward(blob_bottom_vec_, blob_top_vec_);
    EXPECT_NEAR(loss / count, blob_top_loss_->cpu_data()[0], 1e-3);
}

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestForwardIgnoreLabel) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_param()->set_tree(tree_file_name_);
    layer_param.mutable_loss_param()->set_normalize(false);
    layer_param.mutable_loss_param()->set_ignore_label(-1);

    Blob<Dtype> label_orig(blob_bottom_label_->shape());
    label_orig.CopyFrom(*blob_bottom_label_);

    // First, compute the loss with all labels
    scoped_ptr<SoftmaxTreeWithLossLayer<Dtype>> layer(new SoftmaxTreeWithLossLayer<Dtype>(layer_param));
    layer->SetUp(blob_bottom_vec_, blob_top_vec_);
    layer->Forward(blob_bottom_vec_, blob_top_vec_);
    Dtype full_loss = blob_top_loss_->cpu_data()[0];
    // Now, accumulate the loss, ignoring all but one in each turn.
    Dtype accum_loss = 0;
    for (int label = 0; label < 11; ++label) {
        Dtype* label_data = blob_bottom_label_->mutable_cpu_data();
        for (int i = 0; i < blob_bottom_label_->count(); ++i) {
            if (label_data[i] != label)
                label_data[i] = -1;
        }
        layer->SetUp(blob_bottom_vec_, blob_top_vec_);
        layer->Forward(blob_bottom_vec_, blob_top_vec_);
        accum_loss += blob_top_loss_->cpu_data()[0];

        // Revert back
        blob_bottom_label_->CopyFrom(label_orig);
    }
    EXPECT_NEAR(full_loss, accum_loss, 1e-4);
}

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestGradientIgnoreLabel) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_param()->set_tree(tree_file_name_);
    // labels are in {0, ..., 10}, and 0 is root, so we'll ignore less than a tenth of them
    layer_param.mutable_loss_param()->set_ignore_label(-1);
    Dtype* label_data = blob_bottom_label_->mutable_cpu_data();
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
        if (label_data[i] == 0)
            label_data[i] = -1;
    }
    SoftmaxTreeWithLossLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    checker.CheckGradientExhaustive(&layer, blob_bottom_vec_, blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestGradientUnnormalized) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_param()->set_tree(tree_file_name_);
    layer_param.mutable_loss_param()->set_normalize(false);
    SoftmaxTreeWithLossLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    checker.CheckGradientExhaustive(&layer, blob_bottom_vec_, blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestForwardWithObjectness) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_loss_param()->set_with_objectness(true);
    layer_param.mutable_softmaxtree_param()->set_tree(tree_file_name_);
    layer_param.mutable_loss_param()->set_normalization(LossParameter_NormalizationMode_NONE);
    layer_param.add_loss_weight(1);
    layer_param.add_loss_weight(0);
    // Need shared probability blob at the top
    blob_top_vec_.push_back(blob_top_prob_);
    blob_bottom_vec_.push_back(blob_bottom_objectness_);

    scoped_ptr<SoftmaxTreeWithLossLayer<Dtype>> layer(new SoftmaxTreeWithLossLayer<Dtype>(layer_param));
    layer->SetUp(blob_bottom_vec_, blob_top_vec_);
    layer->Forward(blob_bottom_vec_, blob_top_vec_);
    Dtype loss = 0;
    int count = 0;
    for (int i = 0; i < blob_bottom_data_->num(); ++i) {
        const int label_value_orig = static_cast<int>(blob_bottom_label_->data_at(i, 0, 0, 0));
        EXPECT_GE(label_value_orig, 0);
        EXPECT_LT(label_value_orig, blob_top_prob_->shape(1));
        double max_prob = -1;
        int max_k = 0;
        int max_l = 0;

        for (int k = 0; k < blob_bottom_data_->height(); ++k) {
            for (int l = 0; l < blob_bottom_data_->width(); ++l) {
                double p = blob_bottom_objectness_->data_at(i, 0, k, l);
                int label_value = label_value_orig;
                while (label_value >= 0) {
                    p *= blob_top_prob_->data_at(i, label_value, k, l);
                    label_value = parent_[label_value];
                }
                if (p > max_prob) {
                    max_prob = p;
                    max_k = k;
                    max_l = l;
                }
            }
        }
        int label_value = label_value_orig;
        while (label_value >= 0) {
            loss -= log(std::max(blob_top_prob_->data_at(i, label_value, max_k, max_l), Dtype(FLT_MIN)));
            ++count;
            label_value = parent_[label_value];
        }
    }
    EXPECT_NEAR(loss, blob_top_loss_->cpu_data()[0], 1e-3);

    layer_param.mutable_loss_param()->set_normalization(LossParameter_NormalizationMode_VALID);
    layer.reset(new SoftmaxTreeWithLossLayer<Dtype>(layer_param));
    layer->SetUp(blob_bottom_vec_, blob_top_vec_);
    layer->Forward(blob_bottom_vec_, blob_top_vec_);
    EXPECT_NEAR(loss / count, blob_top_loss_->cpu_data()[0], 1e-3);
}

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestForwardWithObjectnessPerBatchLabels) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_loss_param()->set_with_objectness(true);
    layer_param.mutable_softmaxtree_param()->set_tree(tree_file_name_);
    layer_param.mutable_loss_param()->set_normalization(LossParameter_NormalizationMode_NONE);
    layer_param.add_loss_weight(1);
    layer_param.add_loss_weight(0);
    // Need shared probability blob at the top
    blob_top_vec_.push_back(blob_top_prob_);
    blob_bottom_vec_.push_back(blob_bottom_objectness_);

    blob_bottom_vec_.clear();
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_label_perbatch_);
    blob_bottom_vec_.push_back(blob_bottom_objectness_);

    scoped_ptr<SoftmaxTreeWithLossLayer<Dtype>> layer(new SoftmaxTreeWithLossLayer<Dtype>(layer_param));
    layer->SetUp(blob_bottom_vec_, blob_top_vec_);
    layer->Forward(blob_bottom_vec_, blob_top_vec_);
    Dtype loss = 0;
    int count = 0;
    for (int i = 0; i < blob_bottom_data_->num(); ++i) {
        const int label_value_orig = static_cast<int>(blob_bottom_label_perbatch_->data_at({ i, 0 }));
        EXPECT_GE(label_value_orig, 0);
        EXPECT_LT(label_value_orig, blob_top_prob_->shape(1));
        double max_prob = -1;
        int max_k = 0;
        int max_l = 0;

        for (int k = 0; k < blob_bottom_data_->height(); ++k) {
            for (int l = 0; l < blob_bottom_data_->width(); ++l) {
                double p = blob_bottom_objectness_->data_at(i, 0, k, l);
                int label_value = label_value_orig;
                while (label_value >= 0) {
                    p *= blob_top_prob_->data_at(i, label_value, k, l);
                    label_value = parent_[label_value];
                }
                if (p > max_prob) {
                    max_prob = p;
                    max_k = k;
                    max_l = l;
                }
            }
        }
        int label_value = label_value_orig;
        while (label_value >= 0) {
            loss -= log(std::max(blob_top_prob_->data_at(i, label_value, max_k, max_l), Dtype(FLT_MIN)));
            ++count;
            label_value = parent_[label_value];
        }
    }
    EXPECT_NEAR(loss, blob_top_loss_->cpu_data()[0], 1e-3);

    layer_param.mutable_loss_param()->set_normalization(LossParameter_NormalizationMode_VALID);
    layer.reset(new SoftmaxTreeWithLossLayer<Dtype>(layer_param));
    layer->SetUp(blob_bottom_vec_, blob_top_vec_);
    layer->Forward(blob_bottom_vec_, blob_top_vec_);
    EXPECT_NEAR(loss / count, blob_top_loss_->cpu_data()[0], 1e-3);
}

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestGradientWithObjectness) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_loss_param()->set_with_objectness(true);
    layer_param.mutable_softmaxtree_param()->set_tree(tree_file_name_);
    layer_param.add_loss_weight(3);
    SoftmaxTreeWithLossLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    blob_bottom_vec_.push_back(blob_bottom_objectness_);
    checker.CheckGradientExhaustive(&layer, blob_bottom_vec_, blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestGradientWithObjectnessUnnormalized) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_loss_param()->set_with_objectness(true);
    layer_param.mutable_softmaxtree_param()->set_tree(tree_file_name_);
    layer_param.mutable_loss_param()->set_normalize(false);
    SoftmaxTreeWithLossLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    blob_bottom_vec_.push_back(blob_bottom_objectness_);
    checker.CheckGradientExhaustive(&layer, blob_bottom_vec_, blob_top_vec_, 0);
}

}  // namespace caffe
