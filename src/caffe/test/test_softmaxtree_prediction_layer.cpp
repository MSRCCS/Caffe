#include <cmath>
#include <vector>
#include <cfloat>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/softmaxtree_prediction_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class SoftmaxTreePredictionLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;
protected:
    SoftmaxTreePredictionLayerTest() :
        tree_file_name_(CMAKE_SOURCE_DIR "caffe/test/test_data/11n_4g.tree"),
        group_size_{ 4, 2, 3, 2 },
        group_offset_{ 0, 4, 6, 9 },
        child_{ 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1 },
        child_size_{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        parent_{ -1, -1, -1, -1, 0, 0, 4, 4, 4, 8, 8 },
        root_size_(0),
        blob_bottom_(new Blob<Dtype>(2, 11, 2, 3)),
        blob_top_prob_(new Blob<Dtype>()) {
        Caffe::set_random_seed(9658361);

        blob_bottom_vec_.push_back(blob_bottom_);
        blob_top_vec_.push_back(blob_top_prob_);
        FillBottom();
    }
    virtual ~SoftmaxTreePredictionLayerTest() {
        delete blob_bottom_;
        delete blob_top_prob_;
    }
    void FillBottom() {
        // fill the values
        FillerParameter filler_param;
        filler_param.set_min(0);
        filler_param.set_max(1);
        UniformFiller<Dtype> filler(filler_param);
        filler.Fill(blob_bottom_);
    }
    void Initialize(vector<int> parent, 
                    vector<int> group_size,
                    vector<int> child,
                    vector<int> child_size,
                    int root_size,
                    const char* tree_file_name) {
        ASSERT_STRNE(tree_file_name, NULL);
        ASSERT_EQ(parent.size(), child.size());
        ASSERT_EQ(parent.size(), child_size.size());

        tree_file_name_ = tree_file_name;
        parent_ = parent;
        group_size_ = group_size;
        child_ = child;
        child_size_ = child_size;
        root_size_ = root_size;
        
        group_offset_.clear();
        int n = 0;
        for (auto g : group_size_) {
            group_offset_.emplace_back(n);
            n += g;
        }

        ASSERT_EQ(n, parent.size());
        blob_bottom_->Reshape(2, n, 2, 3);
        FillBottom();
    }
    bool predict_tree(Dtype threshold, int i, int k, int l, int g, double p) {
        int argmax = 0;
        // Tree search
        Dtype maxval = -FLT_MAX;
        {
            auto offset = group_offset_[g];
            auto size = group_size_[g];
            for (int j = 0; j < size; ++j) {
                Dtype prob = blob_bottom_->data_at(i, offset + j, k, l);
                if (prob > maxval) {
                    argmax = offset + j;
                    EXPECT_LT(argmax, blob_bottom_->shape(1));
                    maxval = prob;
                }
            }
        }
        p *= maxval;
        if (p <= threshold)
            return false;
        g = child_[argmax];
        if (g >= 0) {
            // Recurse to each subgroup
            int sg_count = child_size_[argmax] + 1;
            bool all_subgroups = true;
            for (int sg = 0; sg < sg_count; ++sg) {
                if (!predict_tree(threshold, i, k, l, g + sg, p))
                    all_subgroups = false;
            }
            // if all the child subgroups pass the threshold, do not set the parent anymore
            if (all_subgroups)
                return true;
        }

        EXPECT_NEAR(p, blob_top_prob_->data_at(i, argmax, k, l), 1e-4) <<
            " n: " << i << " c: " << argmax << " h: " << k << " w: " << l;
        return true;
    }
    void TestForward(bool append_max, Dtype threshold) {
        for (int i = 0; i < blob_bottom_->num(); ++i) {
            for (int k = 0; k < blob_bottom_->height(); ++k) {
                for (int l = 0; l < blob_bottom_->width(); ++l) {
                    for (int g = 0; g < root_size_ + 1; g++) {
                        predict_tree(threshold, 
                                     i, k, l, g,
                                     1);
                    }
                    if (append_max) {
                        auto channels = blob_top_prob_->channels() - 1;
                        Dtype maxval = -FLT_MAX;
                        for (int j = 0; j < channels; ++j) {
                            Dtype prob = blob_top_prob_->data_at(i, j, k, l);
                            if (prob > maxval)
                                maxval = prob;
                        }
                        EXPECT_FLOAT_EQ(maxval, blob_top_prob_->data_at(i, channels, k, l)) <<
                            " n: " << i << " c: " << channels << " h: " << k << " w: " << l;
                    }
                }
            }
        }
    }
    void TestForwardNoSubgroups(bool append_max, Dtype threshold) {
        // Simple test with no subgroups
        for (int i = 0; i < blob_bottom_->num(); ++i) {
            for (int k = 0; k < blob_bottom_->height(); ++k) {
                for (int l = 0; l < blob_bottom_->width(); ++l) {
                    int g = 0; // start from the root
                    double p = 1;
                    int parent_argmax = 0;
                    Dtype parent_p = 0;
                    int argmax = 0;
                    // Tree search
                    do {
                        auto offset = group_offset_[g];
                        auto size = group_size_[g];
                        Dtype maxval = -FLT_MAX;
                        for (int j = 0; j < size; ++j) {
                            Dtype prob = blob_bottom_->data_at(i, offset + j, k, l);
                            if (prob > maxval) {
                                argmax = offset + j;
                                EXPECT_LT(argmax, blob_bottom_->shape(1));
                                maxval = prob;
                            }
                        }
                        p *= maxval;
                        g = child_[argmax];
                        if (p <= threshold) {
                            argmax = parent_argmax;
                            p = parent_p;
                            break;
                        }
                        parent_p = p;
                        parent_argmax = argmax;
                    } while (g > 0);

                    EXPECT_NEAR(p, blob_top_prob_->data_at(i, argmax, k, l), 1e-4);
                    if (append_max) {
                        EXPECT_FLOAT_EQ(p, blob_top_prob_->data_at(i, blob_top_prob_->channels() - 1, k, l));
                    }
                }
            }
        }
    }
    string tree_file_name_;
    vector<int> group_size_;
    vector<int> group_offset_;
    vector<int> child_;
    vector<int> child_size_;
    vector<int> parent_;
    int root_size_;
    Blob<Dtype>* const blob_bottom_;
    Blob<Dtype>* const blob_top_prob_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SoftmaxTreePredictionLayerTest, TestDtypesAndDevices);

TYPED_TEST(SoftmaxTreePredictionLayerTest, TestForward) {
    typedef typename TypeParam::Dtype Dtype;
    this->Initialize({ -1, -1, -1, -1, 0, 0, 4, 4, 4, 8, 8, 1, 1, 1, 1 },
                     { 2, 2, 2, 3, 2, 2, 2 },
                     { 2, 5, -1, -1, 3, -1, -1, -1, 4, -1, -1, -1, -1, -1, -1 },
                     { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                     1,
                     CMAKE_SOURCE_DIR "caffe/test/test_data/15n_5g_7s.tree");

    const Dtype kThreshold = 0.5;

    LayerParameter layer_param;
    layer_param.mutable_softmaxtreeprediction_param()->set_tree(this->tree_file_name_);
    layer_param.mutable_softmaxtreeprediction_param()->set_threshold(kThreshold);
    scoped_ptr<SoftmaxTreePredictionLayer<Dtype>> layer(new SoftmaxTreePredictionLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    ASSERT_EQ(this->blob_top_prob_->channels(), this->blob_bottom_->channels() + 1);
    this->TestForward(true, kThreshold);

    layer_param.mutable_softmaxtreeprediction_param()->set_append_max(false);
    layer.reset(new SoftmaxTreePredictionLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    ASSERT_EQ(this->blob_top_prob_->count(), this->blob_bottom_->count());
    this->TestForward(false, kThreshold);
}

TYPED_TEST(SoftmaxTreePredictionLayerTest, TestForwardNoThreshold) {
    typedef typename TypeParam::Dtype Dtype;
    const Dtype kThreshold = -1;
    LayerParameter layer_param;
    layer_param.mutable_softmaxtreeprediction_param()->set_threshold(kThreshold);
    layer_param.mutable_softmaxtreeprediction_param()->set_tree(this->tree_file_name_);
    scoped_ptr<SoftmaxTreePredictionLayer<Dtype>> layer(new SoftmaxTreePredictionLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    ASSERT_EQ(layer->StackSize(), 1);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    ASSERT_EQ(this->blob_top_prob_->channels(), this->blob_bottom_->channels() + 1);
    this->TestForwardNoSubgroups(true, kThreshold);

    this->Initialize({ -1, -1, -1, -1, 0, 0, 4, 4, 4, 8, 8, 1, 1, 1, 1 },
                     { 2, 2, 2, 3, 2, 2, 2 },
                     { 2, 5, -1, -1, 3, -1, -1, -1, 4, -1, -1, -1, -1, -1, -1 },
                     { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
                     1,
                     CMAKE_SOURCE_DIR "caffe/test/test_data/15n_5g_7s.tree");

    layer_param.mutable_softmaxtreeprediction_param()->set_tree(this->tree_file_name_);
    layer.reset(new SoftmaxTreePredictionLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    ASSERT_EQ(layer->StackSize(), 2);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    ASSERT_EQ(this->blob_top_prob_->channels(), this->blob_bottom_->channels() + 1);
    this->TestForward(true, kThreshold);
}

TYPED_TEST(SoftmaxTreePredictionLayerTest, TestForwardNoSubgroups) {
    typedef typename TypeParam::Dtype Dtype;

    const Dtype kThreshold = 0.5;

    LayerParameter layer_param;
    layer_param.mutable_softmaxtreeprediction_param()->set_tree(this->tree_file_name_);
    layer_param.mutable_softmaxtreeprediction_param()->set_threshold(kThreshold);
    scoped_ptr<SoftmaxTreePredictionLayer<Dtype>> layer(new SoftmaxTreePredictionLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    ASSERT_EQ(this->blob_top_prob_->channels(), this->blob_bottom_->channels() + 1);
    this->TestForwardNoSubgroups(true, kThreshold);

    layer_param.mutable_softmaxtreeprediction_param()->set_append_max(false);
    layer.reset(new SoftmaxTreePredictionLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    ASSERT_EQ(this->blob_top_prob_->count(), this->blob_bottom_->count());
    this->TestForwardNoSubgroups(false, kThreshold);
}

}  // namespace caffe
