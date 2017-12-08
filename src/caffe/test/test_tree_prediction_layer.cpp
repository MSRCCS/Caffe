#include <cmath>
#include <vector>
#include <cfloat>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/tree_prediction_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class TreePredictionLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;
protected:
    TreePredictionLayerTest() : 
        tree_file_name_(CMAKE_SOURCE_DIR "caffe/test/test_data/11n_4g.tree"),
        map_file_name_(CMAKE_SOURCE_DIR "caffe/test/test_data/label_map.txt"),
        group_size_{ 4, 2, 3, 2 },
        group_offset_{ 0, 4, 6, 9 },
        child_{ 1, -1, -1, -1, 2, -1, -1, -1, 3, -1, -1 },
        parent_{ -1, -1, -1, -1, 0, 0, 4, 4, 4, 8, 8 },
        label_map_{ 0, 1, 4, 6, 8, 9, 10 },
        blob_bottom_(new Blob<Dtype>(2, 11, 2, 3)),
        blob_top_argmax_(new Blob<Dtype>()), 
        blob_top_prob_(new Blob<Dtype>()),
        blob_top_prob_map_(new Blob<Dtype>()),
        blob_top_max_(new Blob<Dtype>()) {
        // fill the values
        FillerParameter filler_param;
        filler_param.set_min(0);
        filler_param.set_max(1);
        UniformFiller<Dtype> filler(filler_param);
        filler.Fill(blob_bottom_);
        blob_bottom_vec_.push_back(blob_bottom_);
        blob_top_vec_.push_back(blob_top_argmax_);
    }
    void MapForward(bool with_max = false) {
        for (int i = 0; i < blob_bottom_->num(); ++i) {
            for (int k = 0; k < blob_bottom_->height(); ++k) {
                for (int l = 0; l < blob_bottom_->width(); ++l) {
                    int argmax = 0;
                    Dtype maxval = -FLT_MAX;

                    // Check hierarchical probabilities
                    for (int j = 0; j < label_map_.size(); ++j) {
                        int label_value = label_map_[j];
                        EXPECT_GE(label_value, 0);
                        EXPECT_LT(label_value, blob_bottom_->shape(1));

                        double p = 1;
                        while (label_value >= 0) {
                            p *= blob_bottom_->data_at(i, label_value, k, l);
                            label_value = parent_[label_value];
                        }

                        EXPECT_NEAR(p, blob_top_prob_map_->data_at(i, j, k, l), 1e-4);

                        if (p > maxval) {
                            argmax = label_map_[j];
                            maxval = p;
                        }
                    }
                    // Check argmax
                    EXPECT_EQ(argmax, blob_top_argmax_->data_at(i, 0, k, l));
                    if(with_max)
                        EXPECT_FLOAT_EQ(maxval, blob_top_max_->data_at(i, 0, k, l));
                }
            }
        }
    }
    virtual ~TreePredictionLayerTest() {
        delete blob_bottom_; 
        delete blob_top_argmax_;
        delete blob_top_prob_;
        delete blob_top_prob_map_;
        delete blob_top_max_;
    }
    string tree_file_name_;
    string map_file_name_;
    vector<int> group_size_;
    vector<int> group_offset_;
    vector<int> child_;
    vector<int> parent_;
    vector<int> label_map_;
    Blob<Dtype>* const blob_bottom_;
    Blob<Dtype>* const blob_top_argmax_;
    Blob<Dtype>* const blob_top_prob_;
    Blob<Dtype>* const blob_top_prob_map_;
    Blob<Dtype>* const blob_top_max_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(TreePredictionLayerTest, TestDtypesAndDevices);

TYPED_TEST(TreePredictionLayerTest, TestForward) {
    typedef typename TypeParam::Dtype Dtype;

    const Dtype kThreshold = 0.5;

    LayerParameter layer_param;
    layer_param.mutable_treeprediction_param()->set_tree(this->tree_file_name_);
    layer_param.mutable_treeprediction_param()->set_threshold(kThreshold);
    TreePredictionLayer<Dtype> layer(layer_param);
    this->blob_top_vec_.push_back(this->blob_top_prob_);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < this->blob_bottom_->num(); ++i) {
        for (int k = 0; k < this->blob_bottom_->height(); ++k) {
            for (int l = 0; l < this->blob_bottom_->width(); ++l) {
                int g = 0; // start from the root
                double p = 1;
                int parent_argmax = 0;
                Dtype parent_p = 0;
                int argmax = 0;
                // Tree search
                do {
                    auto offset = this->group_offset_[g];
                    auto size = this->group_size_[g];
                    Dtype maxval = -FLT_MAX;
                    for (int j = 0; j < size; ++j) {
                        Dtype prob = this->blob_bottom_->data_at(i, offset + j, k, l);
                        if (prob > maxval) {
                            argmax = offset + j;
                            EXPECT_LT(argmax, this->blob_bottom_->shape(1));
                            maxval = prob;
                        }
                    }
                    p *= maxval;
                    g = this->child_[argmax];
                    if (p <= kThreshold) {
                        argmax = parent_argmax;
                        p = parent_p;
                        break;
                    }
                    parent_p = p;
                    parent_argmax = argmax;
                } while (g > 0);

                EXPECT_EQ(argmax, this->blob_top_argmax_->data_at(i, 0, k, l));
                EXPECT_NEAR(p, this->blob_top_prob_->data_at(i, 0, k, l), 1e-4);
            }
        }
    }
}

TYPED_TEST(TreePredictionLayerTest, TestForwardWithMap) {
    typedef typename TypeParam::Dtype Dtype;
    LayerParameter layer_param;
    layer_param.mutable_treeprediction_param()->set_tree(this->tree_file_name_);
    layer_param.mutable_treeprediction_param()->set_map(this->map_file_name_);

    TreePredictionLayer<Dtype> layer(layer_param);
    this->blob_top_vec_.push_back(this->blob_top_prob_map_);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    this->MapForward();

    this->blob_top_vec_.push_back(this->blob_top_max_);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    this->MapForward(true);
}

}  // namespace caffe
