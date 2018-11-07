#include <vector>
#include <numeric>
#include <cfloat>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/layers/yolo_eval_compat_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class YoloEvalCompatLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;
protected:
    YoloEvalCompatLayerTest() :
        blob_conf1_(new Blob<Dtype>(2, 5, 5, 3)),
        blob_conf2_(new Blob<Dtype>(2, 6, 5, 3)),
        blob_conf3_(new Blob<Dtype>({ 2, 2, 6, 2, 4 })),
        blob_top_conf_(new Blob<Dtype>()) {
        Caffe::set_random_seed(777);

        FillerParameter filler_param;
        filler_param.set_min(0);
        filler_param.set_max(1);
        UniformFiller<Dtype> filler(filler_param);
        filler.Fill(blob_conf1_);
        filler.Fill(blob_conf2_);
        filler.Fill(blob_conf3_);

        blob_top_vec_.push_back(blob_top_conf_);
    }
    void Test(bool move_axis, bool append_max=false) {
        int sum_classes = 0;
        int sum_inner_num = 0;
        std::map<std::pair<int, int>, vector<int>> chan_map;
        for (int i = 0; i < blob_bottom_vec_.size(); ++i) {
            auto& bottom = blob_bottom_vec_[i];
            auto classes = bottom->shape(1);
            auto inner_num = bottom->count(2);
            if (!append_max)
                classes--;
            for (int c = 0; c < classes; ++c) {
                for (int s = 0; s < inner_num; ++s) {
                    chan_map[{c + sum_classes, s + sum_inner_num}] = {i, c, s};
                }
            }
            sum_classes += classes;
            sum_inner_num += inner_num;
        }
        if (!append_max) {
            // objectness column is concatenated at the end
            sum_inner_num = 0;
            for (int i = 0; i < blob_bottom_vec_.size(); ++i) {
                auto& bottom = blob_bottom_vec_[i];
                auto classes = bottom->shape(1);
                auto inner_num = bottom->count(2);
                auto c = classes - 1;
                for (int s = 0; s < inner_num; ++s) {
                    chan_map[{sum_classes, s + sum_inner_num}] = { i, c, s };
                }
                sum_inner_num += inner_num;
            }
        }
        int channels = sum_classes + 1;
        auto outer_num = blob_conf1_->shape(0);
        EXPECT_EQ(blob_top_conf_->count(), outer_num * channels * sum_inner_num)
            << "bottoms: " << blob_bottom_vec_.size();
        for (int n = 0; n < outer_num; ++n) {
            for (int s = 0; s < sum_inner_num; ++s) {
                Dtype conf;
                Dtype maxval = -FLT_MAX;
                int out_offset;
                for (int c = 0; c < sum_classes; ++c) {
                    if (move_axis) {
                        conf = blob_top_conf_->data_at({ n, s, c });
                        out_offset = blob_top_conf_->offset({ n, s, c });
                    } else {
                        conf = blob_top_conf_->data_at({ n, c, s });
                        out_offset = blob_top_conf_->offset({ n, c, s });
                    }
                    if (conf > maxval)
                        maxval = conf;
                    auto it = chan_map.find({ c, s });
                    if (it == chan_map.end()) {
                        EXPECT_FLOAT_EQ(conf, 0)
                            << "bottoms: " << blob_bottom_vec_.size() << " n: " << n << " oc: " << c << " os: " << s;
                        continue;
                    }
                    auto& ics = it->second;
                    auto i = ics[0];
                    auto bc = ics[1];
                    auto bs = ics[2];
                    auto& bottom = blob_bottom_vec_[i];
                    auto classes = bottom->shape(1);
                    auto inner_num = bottom->count(2);
                    auto offset = (n * classes + bc) * inner_num + bs;
                    EXPECT_FLOAT_EQ(conf, bottom->cpu_data()[offset]) 
                        << "bottoms: " << blob_bottom_vec_.size()
                        << " i: " << i << " n: " << n << " c: " << bc << " s: " << bs 
                        << " oc: " << c << " os: " << s 
                        << " bottom_off: " << offset << " output_off: " << out_offset;
                }
                // objectness test
                auto c = sum_classes;
                if (move_axis) {
                    conf = blob_top_conf_->data_at({ n, s, c });
                    out_offset = blob_top_conf_->offset({ n, s, sum_classes });
                } else {
                    conf = blob_top_conf_->data_at({ n, c, s });
                    out_offset = blob_top_conf_->offset({ n, sum_classes, s });
                }
                if (append_max) {
                    EXPECT_FLOAT_EQ(conf, maxval)
                        << "bottoms: " << blob_bottom_vec_.size()
                        << " n: " << n 
                        << " oc: " << c << " os: " << s
                        << " output_off: " << out_offset;
                } else {
                    auto it = chan_map.find({ c, s });
                    if (it == chan_map.end()) {
                        maxval = 0;
                    } else {
                        auto& ics = it->second;
                        auto& bottom = blob_bottom_vec_[ics[0]];
                        auto classes = bottom->shape(1);
                        EXPECT_EQ(ics[1], classes - 1)
                            << "bottoms: " << blob_bottom_vec_.size()
                            << " n: " << n
                            << " oc: " << c << " os: " << s
                            << " output_off: " << out_offset;
                        auto inner_num = bottom->count(2);
                        auto offset = (n * classes + ics[1]) * inner_num + ics[2];
                        maxval = bottom->cpu_data()[offset];
                    }
                    EXPECT_FLOAT_EQ(conf, maxval)
                        << "bottoms: " << blob_bottom_vec_.size()
                        << " n: " << n
                        << " oc: " << c << " os: " << s
                        << " output_off: " << out_offset;
                }
            }
        }
    }
    virtual ~YoloEvalCompatLayerTest() {
        delete blob_conf1_;
        delete blob_conf2_;
        delete blob_conf3_;
        delete blob_top_conf_;
    }
    Blob<Dtype>* const blob_conf1_;
    Blob<Dtype>* const blob_conf2_;
    Blob<Dtype>* const blob_conf3_;
    Blob<Dtype>* const blob_top_conf_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(YoloEvalCompatLayerTest, TestDtypesAndDevices);

TYPED_TEST(YoloEvalCompatLayerTest, TestForward) {
    typedef typename TypeParam::Dtype Dtype;

    LayerParameter layer_param;
    layer_param.mutable_yoloevalcompat_param()->set_append_max(false);
    layer_param.mutable_yoloevalcompat_param()->set_move_axis(true);
    YoloEvalCompatLayer<Dtype> layer(layer_param);

    this->blob_bottom_vec_.push_back(this->blob_conf1_);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    this->Test(true);

    this->blob_bottom_vec_.push_back(this->blob_conf2_);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    this->Test(true);

    this->blob_bottom_vec_.push_back(this->blob_conf3_);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    this->Test(true);
}

TYPED_TEST(YoloEvalCompatLayerTest, TestForwardNoMove) {
    typedef typename TypeParam::Dtype Dtype;

    LayerParameter layer_param;
    layer_param.mutable_yoloevalcompat_param()->set_append_max(false);
    layer_param.mutable_yoloevalcompat_param()->set_move_axis(false);
    YoloEvalCompatLayer<Dtype> layer(layer_param);

    this->blob_bottom_vec_.push_back(this->blob_conf1_);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    this->Test(false);

    this->blob_bottom_vec_.push_back(this->blob_conf2_);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    this->Test(false);

    this->blob_bottom_vec_.push_back(this->blob_conf3_);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    this->Test(false);
}

TYPED_TEST(YoloEvalCompatLayerTest, TestForwardAppend) {
    typedef typename TypeParam::Dtype Dtype;

    LayerParameter layer_param;
    layer_param.mutable_yoloevalcompat_param()->set_append_max(true);
    layer_param.mutable_yoloevalcompat_param()->set_move_axis(true);
    YoloEvalCompatLayer<Dtype> layer(layer_param);

    this->blob_bottom_vec_.push_back(this->blob_conf1_);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    this->Test(true, true);

    this->blob_bottom_vec_.push_back(this->blob_conf2_);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    this->Test(true, true);

    this->blob_bottom_vec_.push_back(this->blob_conf3_);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    this->Test(true, true);
}

TYPED_TEST(YoloEvalCompatLayerTest, TestForwardAppendNoMove) {
    typedef typename TypeParam::Dtype Dtype;

    LayerParameter layer_param;
    layer_param.mutable_yoloevalcompat_param()->set_append_max(true);
    layer_param.mutable_yoloevalcompat_param()->set_move_axis(false);
    YoloEvalCompatLayer<Dtype> layer(layer_param);

    this->blob_bottom_vec_.push_back(this->blob_conf1_);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    this->Test(false, true);

    this->blob_bottom_vec_.push_back(this->blob_conf2_);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    this->Test(false, true);

    this->blob_bottom_vec_.push_back(this->blob_conf3_);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    this->Test(false, true);
}


}
