#include <vector>
#include <numeric>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/region_common.hpp"

#include "caffe/layers/nms_filter_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class NMSFilterLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;
private:
    vector<int> ShuffledIndex(int count) {
        vector<int> result(count);
        std::iota(result.begin(), result.end(), 0);
        shuffle(result.begin(), result.end());

        return result;
    }
protected:
    NMSFilterLayerTest() :
        blob_bbs_(new Blob<Dtype>(2, 5, 3, 4)),
        blob_conf_(new Blob<Dtype>(2, 6, 5, 3)),
        blob_conf_one_(new Blob<Dtype>({ 2, 5, 3 })),
        blob_top_conf_(new Blob<Dtype>()) {
        Caffe::set_random_seed(777);

        // Form 2 clusters around (0, 0) and (100, 100)
        auto num_bb = blob_bbs_->count(0, 3);
        vector<float> width(num_bb);
        vector<float> height(num_bb);
        caffe_rng_uniform(num_bb, 1.0f, 100.0f, &width[0]);
        caffe_rng_uniform(num_bb, 1.0f, 100.0f, &height[0]);
        for (int i = 0; i < num_bb; ++i) {
            auto bb = blob_bbs_->mutable_cpu_data() + i * 4;
            if (caffe_rng_rand() % 2) {
                bb[0] = 0;
                bb[1] = 0;
            } else {
                bb[0] = 100;
                bb[1] = 100;
            }
            bb[2] = width[i];
            bb[3] = height[i];
        }
        blob_bottom_vec_.push_back(blob_bbs_);
        blob_top_vec_.push_back(blob_top_conf_);
    }
    vector<vector<int>> FillSortedUniform(int outer_num, int channels, int inner_num,
                                          int c,
                                          Dtype* data) {
        vector<vector<int>> indices(outer_num);
        int n = 0;
        for (auto& idx : indices) {
            idx = ShuffledIndex(inner_num);
            Dtype val = 1.0;
            for (auto i : idx) {
                data[(n * channels + c) * inner_num + i] = val;
                val -= 1. / inner_num;
            }
            n++;
        }
        return indices;
    }
    void TestOneClass(const vector<vector<int>>& idx,
                      const Dtype* bbs_data,
                      int outer_num, int channels, int inner_num,
                      int c,
                      float thresh,
                      const Dtype* conf_data,
                      const Dtype* top_conf_data) {
        for (int n = 0; n < outer_num; ++n) {
            int zeroed_count = 0; // how many have become zero
            int filtered_count = 0; // how many are filtered
            vector<bool> filtered(inner_num);
            for (int i = 0; i < inner_num; ++i) {
                if (top_conf_data[(n * channels + c) * inner_num + idx[n][i]] == 0) {
                    if (conf_data[(n * channels + c) * inner_num + idx[n][i]] != 0)
                        zeroed_count++;
                    continue;
                }
                auto i_bb = bbs_data + (n * inner_num + idx[n][i]) * 4;
                for (int j = i + 1; j < inner_num; ++j) {
                    auto j_bb = bbs_data + (n * inner_num + idx[n][j]) * 4;
                    Dtype curr_iou = TBoxIou<Dtype>(i_bb[0], i_bb[1], i_bb[2], i_bb[3],
                                                    j_bb[0], j_bb[1], j_bb[2], j_bb[3]);
                    if (curr_iou > thresh) {
                        EXPECT_EQ(top_conf_data[(n * channels + c) * inner_num + idx[n][j]], 0) 
                            << "c: " << c;
                        if (!filtered[j]) {
                            filtered[j] = true;
                            filtered_count++;
                        }
                    }
                }
            }
            EXPECT_EQ(filtered_count, zeroed_count)
                << "c: " << c;
        }
    }
    virtual ~NMSFilterLayerTest() {
        delete blob_bbs_;
        delete blob_conf_;
        delete blob_conf_one_;
        delete blob_top_conf_;
    }
    Blob<Dtype>* const blob_bbs_;
    Blob<Dtype>* const blob_conf_;
    Blob<Dtype>* const blob_conf_one_;
    Blob<Dtype>* const blob_top_conf_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(NMSFilterLayerTest, TestDtypesAndDevices);

TYPED_TEST(NMSFilterLayerTest, TestForwardOne) {
    typedef typename TypeParam::Dtype Dtype;

    const float kNMSThreshold = 0.5;
    int outer_num = this->blob_bbs_->shape(0);
    int inner_num = this->blob_bbs_->count(1, 3);
    int channels = 1;
    auto idx = this->FillSortedUniform(outer_num, channels, inner_num,
                                       0,
                                       this->blob_conf_one_->mutable_cpu_data());

    LayerParameter layer_param;
    layer_param.mutable_nmsfilter_param()->set_threshold(kNMSThreshold);
    layer_param.mutable_nmsfilter_param()->set_classes(0);
    this->blob_bottom_vec_.push_back(this->blob_conf_one_);
    NMSFilterLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    this->TestOneClass(idx,
                       this->blob_bbs_->cpu_data(),
                       outer_num, channels, inner_num,
                       0,
                       kNMSThreshold,
                       this->blob_conf_one_->cpu_data(),
                       this->blob_top_conf_->cpu_data());
}

TYPED_TEST(NMSFilterLayerTest, TestForwardPerClass) {
    typedef typename TypeParam::Dtype Dtype;

    const float kNMSThreshold = 0.5;
    int outer_num = this->blob_bbs_->shape(0);
    int inner_num = this->blob_bbs_->count(1, 3);
    int channels = this->blob_conf_->shape(1);
    EXPECT_GT(channels, 1);
    vector<vector<vector<int>>> indices(channels);
    for (int c = 0; c < channels; ++c) {
        auto idx = this->FillSortedUniform(outer_num, channels, inner_num,
                                           c,
                                           this->blob_conf_->mutable_cpu_data());
        indices[c] = idx;
    }

    LayerParameter layer_param;
    layer_param.mutable_nmsfilter_param()->set_classes(-1); // filter all classes
    layer_param.mutable_nmsfilter_param()->set_threshold(kNMSThreshold);
    this->blob_bottom_vec_.push_back(this->blob_conf_);
    scoped_ptr<NMSFilterLayer<Dtype>> layer(new NMSFilterLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    for (int c = 0; c < channels; ++c) {
        const auto& idx = indices[c];
        this->TestOneClass(idx,
                           this->blob_bbs_->cpu_data(),
                           outer_num, channels, inner_num,
                           c,
                           kNMSThreshold,
                           this->blob_conf_->cpu_data(),
                           this->blob_top_conf_->cpu_data());
    }

    const int kClasses = 3; // filter only the first few classes
    CHECK_LT(kClasses, channels);
    layer_param.mutable_nmsfilter_param()->set_classes(kClasses);
    layer.reset(new NMSFilterLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int c = 0; c < kClasses; ++c) {
        const auto& idx = indices[c];
        this->TestOneClass(idx,
                           this->blob_bbs_->cpu_data(),
                           outer_num, channels, inner_num,
                           c,
                           kNMSThreshold,
                           this->blob_conf_->cpu_data(),
                           this->blob_top_conf_->cpu_data());
    }
    // The rest of the classes must be unfiltered
    for (int n = 0; n < outer_num; ++n) {
        for (int c = kClasses; c < channels; ++c) {
            for (int s = 0; s < inner_num; ++s) {
                auto index = (n * channels + c) * inner_num + s;
                EXPECT_FLOAT_EQ(this->blob_conf_->cpu_data()[index],
                                this->blob_top_conf_->cpu_data()[index]);
            }
        }
    }
}

TYPED_TEST(NMSFilterLayerTest, TestForwardPerClassMiddle) {
    typedef typename TypeParam::Dtype Dtype;

    const float kNMSThreshold = 0.5;
    int outer_num = this->blob_bbs_->shape(0);
    int inner_num = this->blob_bbs_->count(1, 3);
    int channels = this->blob_conf_->shape(1);
    EXPECT_GT(channels, 1);
    vector<vector<vector<int>>> indices(channels);
    for (int c = 0; c < channels; ++c) {
        auto idx = this->FillSortedUniform(outer_num, channels, inner_num,
                                           c,
                                           this->blob_conf_->mutable_cpu_data());
        indices[c] = idx;
    }

    const int kClasses = 3; // filter only the first few classes
    const int kFirstClass = 1; // filter starting this class
    CHECK_LT(kClasses + kFirstClass, channels);

    LayerParameter layer_param;
    layer_param.mutable_nmsfilter_param()->set_classes(kClasses);
    layer_param.mutable_nmsfilter_param()->set_threshold(kNMSThreshold);
    layer_param.mutable_nmsfilter_param()->set_first_class(kFirstClass);
    NMSFilterLayer<Dtype> layer(layer_param);
    this->blob_bottom_vec_.push_back(this->blob_conf_);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    for (int c = kFirstClass; c < kClasses + kFirstClass; ++c) {
        const auto& idx = indices[c];
        this->TestOneClass(idx,
                           this->blob_bbs_->cpu_data(),
                           outer_num, channels, inner_num,
                           c,
                           kNMSThreshold,
                           this->blob_conf_->cpu_data(),
                           this->blob_top_conf_->cpu_data());
    }
    // The rest of the classes must be unfiltered
    for (int n = 0; n < outer_num; ++n) {
        for (int c = 0; c < channels; ++c) {
            if (c < kFirstClass || c >= kClasses + kFirstClass) {
                for (int s = 0; s < inner_num; ++s) {
                    auto index = (n * channels + c) * inner_num + s;
                    EXPECT_FLOAT_EQ(this->blob_conf_->cpu_data()[index],
                                    this->blob_top_conf_->cpu_data()[index]) <<
                        "n: " << n << " c: " << c << " s: " << s;
                }
            }
        }
    }
}

TYPED_TEST(NMSFilterLayerTest, TestForwardPreThreshold) {
    typedef typename TypeParam::Dtype Dtype;

    const float kNMSThreshold = -1;
    const float kPreThreshold = .6;
    int outer_num = this->blob_bbs_->shape(0);
    int inner_num = this->blob_bbs_->count(1, 3);
    int channels = this->blob_conf_->shape(1);
    EXPECT_GT(channels, 1);
    const int kClasses = 3; // filter only the first few classes
    CHECK_LT(kClasses, channels);
    FillerParameter filler_param;
    filler_param.set_min(0);
    filler_param.set_max(1);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_conf_);

    LayerParameter layer_param;
    layer_param.mutable_nmsfilter_param()->set_classes(kClasses);
    layer_param.mutable_nmsfilter_param()->set_threshold(kNMSThreshold);
    layer_param.mutable_nmsfilter_param()->set_pre_threshold(kPreThreshold);
    this->blob_bottom_vec_.push_back(this->blob_conf_);
    NMSFilterLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    for (int n = 0; n < outer_num; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int s = 0; s < inner_num; ++s) {
                auto index = (n * channels + c) * inner_num + s;
                auto p = this->blob_conf_->cpu_data()[index];
                if (c < kClasses && p <= kPreThreshold)
                    EXPECT_FLOAT_EQ(this->blob_top_conf_->cpu_data()[index], 0);
                else
                    EXPECT_FLOAT_EQ(this->blob_conf_->cpu_data()[index],
                                    this->blob_top_conf_->cpu_data()[index]);
            }
        }
    }
}

TYPED_TEST(NMSFilterLayerTest, TestForwardPreThresholdMiddle) {
    typedef typename TypeParam::Dtype Dtype;

    const float kNMSThreshold = -1;
    const float kPreThreshold = .6;
    int outer_num = this->blob_bbs_->shape(0);
    int inner_num = this->blob_bbs_->count(1, 3);
    int channels = this->blob_conf_->shape(1);
    EXPECT_GT(channels, 1);
    const int kClasses = 3; // filter only the first few classes
    const int kFirstClass = 1; // filter starting this class
    CHECK_LT(kClasses + kFirstClass, channels);
    FillerParameter filler_param;
    filler_param.set_min(0);
    filler_param.set_max(1);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_conf_);

    LayerParameter layer_param;
    layer_param.mutable_nmsfilter_param()->set_classes(kClasses);
    layer_param.mutable_nmsfilter_param()->set_first_class(kFirstClass);
    layer_param.mutable_nmsfilter_param()->set_threshold(kNMSThreshold);
    layer_param.mutable_nmsfilter_param()->set_pre_threshold(kPreThreshold);
    this->blob_bottom_vec_.push_back(this->blob_conf_);
    NMSFilterLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    for (int n = 0; n < outer_num; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int s = 0; s < inner_num; ++s) {
                auto index = (n * channels + c) * inner_num + s;
                auto p = this->blob_conf_->cpu_data()[index];
                if (c < kClasses + kFirstClass && c >= kFirstClass && p <= kPreThreshold)
                    EXPECT_FLOAT_EQ(this->blob_top_conf_->cpu_data()[index], 0);
                else
                    EXPECT_FLOAT_EQ(this->blob_conf_->cpu_data()[index],
                                    this->blob_top_conf_->cpu_data()[index]);
            }
        }
    }
}

}
