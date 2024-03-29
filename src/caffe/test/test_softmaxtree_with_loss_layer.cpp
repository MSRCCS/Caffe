#include <cmath>
#include <vector>
#include <cfloat>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/softmaxtree_loss_layer.hpp"
#include "caffe/layers/softmax_loss_layer.hpp"

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
        blob_top_index_(new Blob<Dtype>()),
        blob_top_prob_(new Blob<Dtype>()) {
        blob_bottom_vec_.push_back(blob_bottom_data_);
        blob_bottom_vec_.push_back(blob_bottom_label_);
        blob_top_vec_.push_back(blob_top_loss_);
        FillBottom();
    }
    virtual ~SoftmaxTreeWithLossLayerTest() {
        delete blob_bottom_data_;
        delete blob_bottom_label_;
        delete blob_bottom_label_perbatch_;
        delete blob_bottom_objectness_;
        delete blob_top_loss_;
        delete blob_top_index_;
        delete blob_top_prob_;
    }
    void FillBottom() {
        // fill the values
        int n = blob_bottom_data_->shape(1);
        FillerParameter filler_param;
        filler_param.set_std(10);
        GaussianFiller<Dtype> filler(filler_param);
        filler.Fill(blob_bottom_data_);
        for (int i = 0; i < blob_bottom_label_->count(); ++i) {
            blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % n;
        }
        for (int i = 0; i < blob_bottom_label_perbatch_->count(); ++i) {
            blob_bottom_label_perbatch_->mutable_cpu_data()[i] = caffe_rng_rand() % n;
        }
        FillerParameter uni_filler_param;
        uni_filler_param.set_min(0);
        uni_filler_param.set_max(1);
        UniformFiller<Dtype> uni_filler(uni_filler_param);
        uni_filler.Fill(blob_bottom_objectness_);
    }
    void Initialize(vector<int> parent, const char* tree_file_name) {
        ASSERT_STRNE(tree_file_name, NULL);
        tree_file_name_ = tree_file_name;
        parent_ = parent;
        int n = (int)parent.size();
        blob_bottom_data_->Reshape(5, n, 2, 3);
        FillBottom();
    }
    Dtype TestForward() {
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

        return loss / count;
    }
    Dtype TestForwardWithObjectness(Blob<Dtype>* const blob_label) {
        vector<int> index;
        Dtype loss = 0;
        int count = 0;
        for (int i = 0; i < blob_bottom_data_->num(); ++i) {
            const int label_value_orig = static_cast<int>(blob_label->data_at({ i, 0 }));
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
            index.push_back(max_k * blob_bottom_data_->width() + max_l);
            int label_value = label_value_orig;
            while (label_value >= 0) {
                loss -= log(std::max(blob_top_prob_->data_at(i, label_value, max_k, max_l), Dtype(FLT_MIN)));
                ++count;
                label_value = parent_[label_value];
            }
        }
        EXPECT_NEAR(loss, blob_top_loss_->cpu_data()[0], 1e-3);

        // also test the returned indices
        EXPECT_EQ(index.size(), blob_top_index_->count());
        for (int i = 0; i < index.size(); ++i)
            EXPECT_EQ(index[i], static_cast<int>(blob_top_index_->data_at({ i, 0 })));

        return loss / count;
    }
    Dtype TestForwardIgnoreLabel(SoftmaxTreeWithLossLayer<Dtype>* layer) {
        Blob<Dtype> label_orig(blob_bottom_label_->shape());
        label_orig.CopyFrom(*blob_bottom_label_);

        int n = blob_bottom_data_->shape(1);
        // Now, accumulate the loss, ignoring all but one in each turn.
        Dtype accum_loss = 0;
        for (int label = 0; label < n; ++label) {
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
        return accum_loss;
    }
    string tree_file_name_;
    vector<int> parent_;
    Blob<Dtype>* const blob_bottom_data_;
    Blob<Dtype>* const blob_bottom_label_;
    Blob<Dtype>* const blob_bottom_label_perbatch_;
    Blob<Dtype>* const blob_bottom_objectness_;
    Blob<Dtype>* const blob_top_loss_;
    Blob<Dtype>* const blob_top_index_;
    Blob<Dtype>* const blob_top_prob_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SoftmaxTreeWithLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestGradient) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_param()->set_tree(this->tree_file_name_);
    layer_param.add_loss_weight(3);
    SoftmaxTreeWithLossLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestGradientSubGroups) {
    typedef typename TypeParam::Dtype Dtype;
    this->Initialize({ -1, -1, -1, -1, 0, 0, 4, 4, 4, 8, 8, 1, 1, 1, 1 },
                     CMAKE_SOURCE_DIR "caffe/test/test_data/15n_5g_7s.tree");
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_param()->set_tree(this->tree_file_name_);
    layer_param.add_loss_weight(3);
    SoftmaxTreeWithLossLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestForward) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_param()->set_tree(this->tree_file_name_);
    layer_param.mutable_loss_param()->set_normalization(LossParameter_NormalizationMode_NONE);
    layer_param.add_loss_weight(1);
    layer_param.add_loss_weight(0);
    // Need shared probability blob at the top
    this->blob_top_vec_.push_back(this->blob_top_prob_);

    scoped_ptr<SoftmaxTreeWithLossLayer<Dtype> > layer(new SoftmaxTreeWithLossLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    Dtype avg_loss = this->TestForward();

    layer_param.mutable_loss_param()->set_normalization(LossParameter_NormalizationMode_VALID);
    layer.reset(new SoftmaxTreeWithLossLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_NEAR(avg_loss, this->blob_top_loss_->cpu_data()[0], 1e-3);
}

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestForwardSubGroups) {
    typedef typename TypeParam::Dtype Dtype;
    this->Initialize({ -1, -1, -1, -1, 0, 0, 4, 4, 4, 8, 8, 1, 1, 1, 1 },
                     CMAKE_SOURCE_DIR "caffe/test/test_data/15n_5g_7s.tree");
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_param()->set_tree(this->tree_file_name_);
    layer_param.mutable_loss_param()->set_normalization(LossParameter_NormalizationMode_NONE);
    layer_param.add_loss_weight(1);
    layer_param.add_loss_weight(0);
    // Need shared probability blob at the top
    this->blob_top_vec_.push_back(this->blob_top_prob_);

    scoped_ptr<SoftmaxTreeWithLossLayer<Dtype> > layer(new SoftmaxTreeWithLossLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    Dtype avg_loss = this->TestForward();

    layer_param.mutable_loss_param()->set_normalization(LossParameter_NormalizationMode_VALID);
    layer.reset(new SoftmaxTreeWithLossLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_NEAR(avg_loss, this->blob_top_loss_->cpu_data()[0], 1e-3);
}

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestForwardIgnoreLabel) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_param()->set_tree(this->tree_file_name_);
    layer_param.mutable_loss_param()->set_normalize(false);
    layer_param.mutable_loss_param()->set_ignore_label(-1);

    // First, compute the loss with all labels
    scoped_ptr<SoftmaxTreeWithLossLayer<Dtype>> layer(new SoftmaxTreeWithLossLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    Dtype full_loss = this->blob_top_loss_->cpu_data()[0];
    Dtype accum_loss = this->TestForwardIgnoreLabel(layer.get());
    EXPECT_NEAR(full_loss, accum_loss, 1e-4);
}

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestForwardIgnoreLabelSubGroups) {
    typedef typename TypeParam::Dtype Dtype;
    this->Initialize({ -1, -1, -1, -1, 0, 0, 4, 4, 4, 8, 8, 1, 1, 1, 1 },
                     CMAKE_SOURCE_DIR "caffe/test/test_data/15n_5g_7s.tree");
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_param()->set_tree(this->tree_file_name_);
    layer_param.mutable_loss_param()->set_normalize(false);
    layer_param.mutable_loss_param()->set_ignore_label(-1);

    // First, compute the loss with all labels
    scoped_ptr<SoftmaxTreeWithLossLayer<Dtype>> layer(new SoftmaxTreeWithLossLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    Dtype full_loss = this->blob_top_loss_->cpu_data()[0];
    Dtype accum_loss = this->TestForwardIgnoreLabel(layer.get());
    EXPECT_NEAR(full_loss, accum_loss, 1e-4);
}

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestForwardRoot) {
    typedef typename TypeParam::Dtype Dtype;
    this->Initialize({ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
                     CMAKE_SOURCE_DIR "caffe/test/test_data/15n_1g.tree");
    LayerParameter softmax_loss_param;
    softmax_loss_param.mutable_loss_param()->set_normalize(true);
    softmax_loss_param.mutable_loss_param()->set_ignore_label(-1);
    shared_ptr<SoftmaxWithLossLayer<Dtype>> softmax_loss_layer(new SoftmaxWithLossLayer<Dtype>(softmax_loss_param));
    softmax_loss_layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    softmax_loss_layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    Dtype loss = this->blob_top_loss_->cpu_data()[0];

    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_param()->set_tree(this->tree_file_name_);
    layer_param.mutable_loss_param()->set_normalize(true);
    layer_param.mutable_loss_param()->set_ignore_label(-1);
    scoped_ptr<SoftmaxTreeWithLossLayer<Dtype>> layer(new SoftmaxTreeWithLossLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    Dtype full_loss = this->blob_top_loss_->cpu_data()[0];

    EXPECT_FLOAT_EQ(full_loss, loss);
}

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestForwardRootIgnoreLabel) {
    typedef typename TypeParam::Dtype Dtype;
    this->Initialize({ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
                     CMAKE_SOURCE_DIR "caffe/test/test_data/15n_1g.tree");
    Dtype* label_data = this->blob_bottom_label_->mutable_cpu_data();
    for (int i = 0; i < this->blob_bottom_label_->count() / 4; ++i) {
        label_data[i] = -1;
    }

    LayerParameter softmax_loss_param;
    softmax_loss_param.mutable_loss_param()->set_normalization(LossParameter_NormalizationMode_VALID);
    softmax_loss_param.mutable_loss_param()->set_ignore_label(-1);
    shared_ptr<SoftmaxWithLossLayer<Dtype>> softmax_loss_layer(new SoftmaxWithLossLayer<Dtype>(softmax_loss_param));
    softmax_loss_layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    softmax_loss_layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    Dtype loss = this->blob_top_loss_->cpu_data()[0];

    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_param()->set_tree(this->tree_file_name_);
    layer_param.mutable_loss_param()->set_normalization(LossParameter_NormalizationMode_VALID);
    layer_param.mutable_loss_param()->set_ignore_label(-1);
    scoped_ptr<SoftmaxTreeWithLossLayer<Dtype>> layer(new SoftmaxTreeWithLossLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    Dtype full_loss = this->blob_top_loss_->cpu_data()[0];

    EXPECT_FLOAT_EQ(full_loss, loss);
}

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestForwardRootIgnoreLabelBatchNormalization) {
    typedef typename TypeParam::Dtype Dtype;
    this->Initialize({ -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 },
                     CMAKE_SOURCE_DIR "caffe/test/test_data/15n_1g.tree");
    Dtype* label_data = this->blob_bottom_label_->mutable_cpu_data();
    for (int i = 0; i < this->blob_bottom_label_->count() / 4; ++i) {
        label_data[i] = -1;
    }

    LayerParameter softmax_loss_param;
    softmax_loss_param.mutable_loss_param()->set_normalization(LossParameter_NormalizationMode_BATCH_SIZE);
    softmax_loss_param.mutable_loss_param()->set_ignore_label(-1);
    shared_ptr<SoftmaxWithLossLayer<Dtype>> softmax_loss_layer(new SoftmaxWithLossLayer<Dtype>(softmax_loss_param));
    softmax_loss_layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    softmax_loss_layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    Dtype loss = this->blob_top_loss_->cpu_data()[0];

    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_param()->set_tree(this->tree_file_name_);
    layer_param.mutable_loss_param()->set_normalization(LossParameter_NormalizationMode_BATCH_SIZE);
    layer_param.mutable_loss_param()->set_ignore_label(-1);
    scoped_ptr<SoftmaxTreeWithLossLayer<Dtype>> layer(new SoftmaxTreeWithLossLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    Dtype full_loss = this->blob_top_loss_->cpu_data()[0];

    EXPECT_FLOAT_EQ(full_loss, loss);
}

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestGradientIgnoreLabel) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_param()->set_tree(this->tree_file_name_);
    // labels are in {0, ..., 10}, and 0 is root, so we'll ignore less than a tenth of them
    layer_param.mutable_loss_param()->set_ignore_label(-1);
    Dtype* label_data = this->blob_bottom_label_->mutable_cpu_data();
    for (int i = 0; i < this->blob_bottom_label_->count(); ++i) {
        if (label_data[i] == 0)
            label_data[i] = -1;
    }
    SoftmaxTreeWithLossLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestGradientIgnoreLabelSubGroups) {
    typedef typename TypeParam::Dtype Dtype;
    this->Initialize({ -1, -1, -1, -1, 0, 0, 4, 4, 4, 8, 8, 1, 1, 1, 1 },
                     CMAKE_SOURCE_DIR "caffe/test/test_data/15n_5g_7s.tree");
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_param()->set_tree(this->tree_file_name_);
    // labels are in {0, ..., 15}, and 0 is root, so we'll ignore less than a tenth of them
    layer_param.mutable_loss_param()->set_ignore_label(-1);
    Dtype* label_data = this->blob_bottom_label_->mutable_cpu_data();
    for (int i = 0; i < this->blob_bottom_label_->count(); ++i) {
        if (label_data[i] == 0)
            label_data[i] = -1;
    }
    SoftmaxTreeWithLossLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestGradientUnnormalized) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_param()->set_tree(this->tree_file_name_);
    layer_param.mutable_loss_param()->set_normalize(false);
    SoftmaxTreeWithLossLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestGradientUnnormalizedSubGroups) {
    typedef typename TypeParam::Dtype Dtype;
    this->Initialize({ -1, -1, -1, -1, 0, 0, 4, 4, 4, 8, 8, 1, 1, 1, 1 },
                     CMAKE_SOURCE_DIR "caffe/test/test_data/15n_5g_7s.tree");
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_param()->set_tree(this->tree_file_name_);
    layer_param.mutable_loss_param()->set_normalize(false);
    SoftmaxTreeWithLossLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestForwardWithObjectness) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_loss_param()->set_with_objectness(true);
    layer_param.mutable_softmaxtree_param()->set_tree(this->tree_file_name_);
    layer_param.mutable_loss_param()->set_normalization(LossParameter_NormalizationMode_NONE);
    layer_param.add_loss_weight(1);
    layer_param.add_loss_weight(0);
    layer_param.add_loss_weight(0);
    // Need shared probability blob at the top
    this->blob_top_vec_.push_back(this->blob_top_index_);
    this->blob_top_vec_.push_back(this->blob_top_prob_);
    this->blob_bottom_vec_.push_back(this->blob_bottom_objectness_);
    
    scoped_ptr<SoftmaxTreeWithLossLayer<Dtype>> layer(new SoftmaxTreeWithLossLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    Dtype avg_loss = this->TestForwardWithObjectness(this->blob_bottom_label_);

    layer_param.mutable_loss_param()->set_normalization(LossParameter_NormalizationMode_VALID);
    layer.reset(new SoftmaxTreeWithLossLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_NEAR(avg_loss, this->blob_top_loss_->cpu_data()[0], 1e-3);
}

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestForwardWithObjectnessSubGroups) {
    typedef typename TypeParam::Dtype Dtype;
    this->Initialize({ -1, -1, -1, -1, 0, 0, 4, 4, 4, 8, 8, 1, 1, 1, 1 },
                     CMAKE_SOURCE_DIR "caffe/test/test_data/15n_5g_7s.tree");
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_loss_param()->set_with_objectness(true);
    layer_param.mutable_softmaxtree_param()->set_tree(this->tree_file_name_);
    layer_param.mutable_loss_param()->set_normalization(LossParameter_NormalizationMode_NONE);
    layer_param.add_loss_weight(1);
    layer_param.add_loss_weight(0);
    layer_param.add_loss_weight(0);
    // Need shared probability blob at the top
    this->blob_top_vec_.push_back(this->blob_top_index_);
    this->blob_top_vec_.push_back(this->blob_top_prob_);
    this->blob_bottom_vec_.push_back(this->blob_bottom_objectness_);

    scoped_ptr<SoftmaxTreeWithLossLayer<Dtype>> layer(new SoftmaxTreeWithLossLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    Dtype avg_loss = this->TestForwardWithObjectness(this->blob_bottom_label_);

    layer_param.mutable_loss_param()->set_normalization(LossParameter_NormalizationMode_VALID);
    layer.reset(new SoftmaxTreeWithLossLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_NEAR(avg_loss, this->blob_top_loss_->cpu_data()[0], 1e-3);
}

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestForwardWithObjectnessPerBatchLabels) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_loss_param()->set_with_objectness(true);
    layer_param.mutable_softmaxtree_param()->set_tree(this->tree_file_name_);
    layer_param.mutable_loss_param()->set_normalization(LossParameter_NormalizationMode_NONE);
    layer_param.add_loss_weight(1);
    layer_param.add_loss_weight(0);
    layer_param.add_loss_weight(0);
    // Need shared probability blob at the top
    this->blob_top_vec_.push_back(this->blob_top_index_);
    this->blob_top_vec_.push_back(this->blob_top_prob_);

    this->blob_bottom_vec_.clear();
    this->blob_bottom_vec_.push_back(this->blob_bottom_data_);
    this->blob_bottom_vec_.push_back(this->blob_bottom_label_perbatch_);
    this->blob_bottom_vec_.push_back(this->blob_bottom_objectness_);

    scoped_ptr<SoftmaxTreeWithLossLayer<Dtype>> layer(new SoftmaxTreeWithLossLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    Dtype avg_loss = this->TestForwardWithObjectness(this->blob_bottom_label_perbatch_);

    layer_param.mutable_loss_param()->set_normalization(LossParameter_NormalizationMode_VALID);
    layer.reset(new SoftmaxTreeWithLossLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_NEAR(avg_loss, this->blob_top_loss_->cpu_data()[0], 1e-3);
}

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestForwardWithObjectnessPerBatchLabelsSubGroups) {
    typedef typename TypeParam::Dtype Dtype;
    this->Initialize({ -1, -1, -1, -1, 0, 0, 4, 4, 4, 8, 8, 1, 1, 1, 1 },
                     CMAKE_SOURCE_DIR "caffe/test/test_data/15n_5g_7s.tree");
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_loss_param()->set_with_objectness(true);
    layer_param.mutable_softmaxtree_param()->set_tree(this->tree_file_name_);
    layer_param.mutable_loss_param()->set_normalization(LossParameter_NormalizationMode_NONE);
    layer_param.add_loss_weight(1);
    layer_param.add_loss_weight(0);
    layer_param.add_loss_weight(0);
    // Need shared probability blob at the top
    this->blob_top_vec_.push_back(this->blob_top_index_);
    this->blob_top_vec_.push_back(this->blob_top_prob_);

    this->blob_bottom_vec_.clear();
    this->blob_bottom_vec_.push_back(this->blob_bottom_data_);
    this->blob_bottom_vec_.push_back(this->blob_bottom_label_perbatch_);
    this->blob_bottom_vec_.push_back(this->blob_bottom_objectness_);

    scoped_ptr<SoftmaxTreeWithLossLayer<Dtype>> layer(new SoftmaxTreeWithLossLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    Dtype avg_loss = this->TestForwardWithObjectness(this->blob_bottom_label_perbatch_);

    layer_param.mutable_loss_param()->set_normalization(LossParameter_NormalizationMode_VALID);
    layer.reset(new SoftmaxTreeWithLossLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    EXPECT_NEAR(avg_loss, this->blob_top_loss_->cpu_data()[0], 1e-3);
}

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestGradientWithObjectness) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_loss_param()->set_with_objectness(true);
    layer_param.mutable_softmaxtree_param()->set_tree(this->tree_file_name_);
    layer_param.add_loss_weight(3);
    SoftmaxTreeWithLossLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    this->blob_bottom_vec_.push_back(this->blob_bottom_objectness_);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestGradientWithObjectnessSubGroups) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    this->Initialize({ -1, -1, -1, -1, 0, 0, 4, 4, 4, 8, 8, 1, 1, 1, 1 },
                     CMAKE_SOURCE_DIR "caffe/test/test_data/15n_5g_7s.tree");
    layer_param.mutable_softmaxtree_loss_param()->set_with_objectness(true);
    layer_param.mutable_softmaxtree_param()->set_tree(this->tree_file_name_);
    layer_param.add_loss_weight(3);
    SoftmaxTreeWithLossLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    this->blob_bottom_vec_.push_back(this->blob_bottom_objectness_);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestGradientWithObjectnessUnnormalized) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_loss_param()->set_with_objectness(true);
    layer_param.mutable_softmaxtree_param()->set_tree(this->tree_file_name_);
    layer_param.mutable_loss_param()->set_normalize(false);
    SoftmaxTreeWithLossLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    this->blob_bottom_vec_.push_back(this->blob_bottom_objectness_);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxTreeWithLossLayerTest, TestGradientWithObjectnessUnnormalizedSubGroups) {
    typedef typename TypeParam::Dtype Dtype;
    this->Initialize({ -1, -1, -1, -1, 0, 0, 4, 4, 4, 8, 8, 1, 1, 1, 1 },
                     CMAKE_SOURCE_DIR "caffe/test/test_data/15n_5g_7s.tree");
    LayerParameter layer_param;
    layer_param.mutable_softmaxtree_loss_param()->set_with_objectness(true);
    layer_param.mutable_softmaxtree_param()->set_tree(this->tree_file_name_);
    layer_param.mutable_loss_param()->set_normalize(false);
    SoftmaxTreeWithLossLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 1e-2);
    this->blob_bottom_vec_.push_back(this->blob_bottom_objectness_);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_, 0);
}

}  // namespace caffe
