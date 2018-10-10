#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#include "caffe/region_common.hpp"
#include "caffe/layers/yolo_co_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class YoloCoOccurrenceLayerTest : public MultiDeviceTest<TypeParam> {
    typedef typename TypeParam::Dtype Dtype;
protected:
    YoloCoOccurrenceLayerTest() :
        blob_obj_(new Blob<Dtype>(8, 3, 5, 13)),
        blob_no_obj_(new Blob<Dtype>(8, 3, 5, 13)),
        blob_truth_(new Blob<Dtype>(8, (4 + 1) * 10, 1, 1)),
        blob_bbs_(new Blob<Dtype>({ 8, 3, 5, 13, 4 })),
        blob_pred_(new Blob<Dtype>({ 8, (5 + 1), 3, 5, 13 })),
        gt_target_(new Blob<Dtype>(8, 3, 5, 13)),
        target_no_obj_(new Blob<Dtype>()) {
        Caffe::set_random_seed(9658361);

        blob_bottom_vec_.push_back(blob_obj_);
        blob_bottom_vec_.push_back(blob_no_obj_);
        blob_bottom_vec_.push_back(blob_truth_);
        blob_bottom_vec_.push_back(blob_bbs_);
        blob_bottom_vec_.push_back(blob_pred_);
        blob_top_vec_.push_back(target_no_obj_);
    }
    virtual ~YoloCoOccurrenceLayerTest() {
        delete blob_obj_;
        delete blob_no_obj_;
        delete blob_truth_;
        delete blob_bbs_;
        delete blob_pred_;
        delete gt_target_;
        delete target_no_obj_;
    }
    void FillBottom() {
        // fill the values
        {
            FillerParameter filler_param;
            filler_param.set_min(0.1);
            filler_param.set_max(1);
            UniformFiller<Dtype> filler(filler_param);
            filler.Fill(blob_obj_);
            filler.Fill(blob_pred_);
        }
        {
            FillerParameter filler_param;
            filler_param.set_value(0);
            ConstantFiller<Dtype> filler(filler_param);
            filler.Fill(blob_no_obj_);
            filler.Fill(gt_target_);
        }

        auto num_bb = blob_bbs_->count(0, 4);
        ASSERT_EQ(num_bb, blob_obj_->count());
        vector<float> width(num_bb);
        vector<float> height(num_bb);
        vector<float> x(num_bb);
        vector<float> y(num_bb);

        caffe_rng_uniform(num_bb, 0.1f, 0.5f, &width[0]);
        caffe_rng_uniform(num_bb, 0.1f, 0.5f, &height[0]);
        caffe_rng_uniform(num_bb, 0.25f, 0.75f, &x[0]);
        caffe_rng_uniform(num_bb, 0.25f, 0.75f, &y[0]);

        auto max_gt = blob_truth_->shape(1) / 5;
        auto classes = blob_pred_->shape(1) - 1;

        for (int n = 0; n < blob_obj_->num(); ++n) {
            for (int a = 0; a < blob_obj_->channels(); ++a) {
                for (int h = 0; h < blob_obj_->height(); ++h) {
                    for (int w = 0; w < blob_obj_->width(); ++w) {
                        auto obj_index = blob_obj_->offset(n, a, h, w);
                        auto bb = blob_bbs_->mutable_cpu_data() + obj_index * 4;
                        bb[0] = x[obj_index];
                        bb[1] = y[obj_index];
                        bb[2] = width[obj_index];
                        bb[3] = height[obj_index];
                    }
                }
            }
            // Randomly select ground-truth for each pixel, from predictions
            for (int t = 0; t < max_gt; ++t) {
                int a = caffe_rng_rand() % blob_obj_->channels();
                int h = caffe_rng_rand() % blob_obj_->height();
                int w = caffe_rng_rand() % blob_obj_->width();
                int cls = caffe_rng_rand() % classes;
                auto obj_index = blob_obj_->offset(n, a, h, w);
                auto bb = blob_bbs_->cpu_data() + obj_index * 4;
                auto truth = blob_truth_->mutable_cpu_data() + blob_truth_->offset(n, t * 5, 0, 0);
                truth[0] = bb[0];
                truth[1] = bb[1];
                truth[2] = bb[2];
                truth[3] = bb[3];
                truth[4] = cls;
                // Set no-objectness for ground-truth
                blob_no_obj_->mutable_cpu_data()[obj_index] = blob_obj_->cpu_data()[obj_index];
                gt_target_->mutable_cpu_data()[obj_index] = 1;
            }
        }
    }
    void Initialize(vector<int> comap_class,
                    vector<int> comap_size,
                    vector<int> comap,
                    vector<float> comap_ixr,
                    vector<float> comap_thresh,
                    vector<float> comap_obj_thresh,
                    const char* labelmap_file_name,
                    const char* comap_file_name) {
        ASSERT_STRNE(labelmap_file_name, NULL);
        ASSERT_STRNE(comap_file_name, NULL);
        ASSERT_EQ(comap_class.size(), comap_size.size());
        ASSERT_EQ(comap.size(), comap_ixr.size());
        ASSERT_EQ(comap.size(), comap_thresh.size());
        ASSERT_EQ(comap.size(), comap_obj_thresh.size());


        labelmap_file_name_ = labelmap_file_name;
        comap_file_name_ = comap_file_name;
        comap_class_ = comap_class;
        comap_size_ = comap_size;
        comap_ = comap;
        comap_thresh_ = comap_thresh;
        comap_obj_thresh_ = comap_obj_thresh;
        comap_ixr_ = comap_ixr;

        comap_offset_.clear();
        int n = 0;
        for (auto s : comap_size) {
            comap_offset_.emplace_back(n);
            n += s;
        }

        ASSERT_EQ(comap_class.size(), comap_offset_.size());
        FillBottom();
    }
    void TestForward() {
        auto max_gt = blob_truth_->shape(1) / 5;
        auto classes = blob_pred_->shape(1) - 1;
        auto co_classes = comap_class_.size();
        int co_occured = 0;
        for (int n = 0; n < blob_obj_->num(); ++n) {
            for (int a = 0; a < blob_obj_->channels(); ++a) {
                for (int h = 0; h < blob_obj_->height(); ++h) {
                    for (int w = 0; w < blob_obj_->width(); ++w) {
                        bool found = false;
                        // If this is a ground-truth already, nothing to do
                        if (gt_target_->data_at(n, a, h, w) <= 0) {
                            for (int t = 0; t < max_gt && !found; ++t) {
                                Dtype tx = blob_truth_->data_at(n, t * 5 + 0, 0, 0);
                                if (!tx)
                                    break;
                                Dtype ty = blob_truth_->data_at(n, t * 5 + 1, 0, 0);
                                Dtype tw = blob_truth_->data_at(n, t * 5 + 2, 0, 0);
                                Dtype th = blob_truth_->data_at(n, t * 5 + 3, 0, 0);
                                Dtype cls = blob_truth_->data_at(n, t * 5 + 4, 0, 0);
                                if (tw <= 0.00001 || th <= 0.00001)
                                    continue;
                                Dtype px = blob_bbs_->data_at({ n, a, h, w, 0 });
                                Dtype py = blob_bbs_->data_at({ n, a, h, w, 1 });
                                Dtype pw = blob_bbs_->data_at({ n, a, h, w, 2 });
                                Dtype ph = blob_bbs_->data_at({ n, a, h, w, 3 });
                                if (pw <= 0.00001 || ph <= 0.00001)
                                    continue;
                                auto ix = TBoxIntersection(px, py, pw, ph,
                                                           tx, ty, tw, th);
                                ix /= (pw * ph); // intersection ratio

                                for (int cidx = 0; cidx < co_classes && !found; ++cidx) {
                                    int size = comap_size_[cidx];
                                    int offset = comap_offset_[cidx];
                                    int c = comap_class_[cidx];
                                    Dtype conf = blob_pred_->data_at({ n, c, a, h, w });
                                    Dtype objectness = blob_pred_->data_at({ n, classes, a, h, w });
                                    for (int i = 0; i < size; ++i) {
                                        int co = comap_[offset + i];
                                        if (co != cls)
                                            continue;
                                        auto thresh = comap_thresh_[offset + i];
                                        if (conf < thresh)
                                            break;
                                        auto obj_thresh = comap_obj_thresh_[offset + i];
                                        if (objectness < obj_thresh)
                                            break;
                                        auto ixr_thresh = comap_ixr_[offset + i];
                                        if (ix >= ixr_thresh) {
                                            found = true;
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                        if (found) {
                            EXPECT_FLOAT_EQ(target_no_obj_->data_at(n, a, h, w),
                                            blob_obj_->data_at(n, a, h, w)) <<
                                " n: " << n << " a: " << a << " h: " << h << " w: " << w;
                            co_occured++;
                        } else {
                            EXPECT_FLOAT_EQ(target_no_obj_->data_at(n, a, h, w),
                                            blob_no_obj_->data_at(n, a, h, w)) <<
                                " n: " << n << " a: " << a << " h: " << h << " w: " << w
                                << " is_gt: " << (gt_target_->data_at(n, a, h, w) > 0);
                        }
                    }
                }
            }
        }
        // Make sure random samples were fine
        EXPECT_GT(co_occured, 0) << "No co-occurrence detected";
        EXPECT_LT(co_occured, blob_obj_->count()) << "All flagged as co-occurrence";
    }
    Blob<Dtype>* const blob_obj_;
    Blob<Dtype>* const blob_no_obj_;
    Blob<Dtype>* const blob_truth_;
    Blob<Dtype>* const blob_bbs_;
    Blob<Dtype>* const blob_pred_;
    Blob<Dtype>* const gt_target_;
    Blob<Dtype>* const target_no_obj_;
    vector<Blob<Dtype>*> blob_bottom_vec_;
    vector<Blob<Dtype>*> blob_top_vec_;
    string labelmap_file_name_;
    string comap_file_name_;
    vector<int> comap_class_;
    vector<int> comap_offset_;
    vector<int> comap_size_;
    vector<int> comap_;
    vector<float> comap_thresh_;
    vector<float> comap_obj_thresh_;
    vector<float> comap_ixr_;
};

TYPED_TEST_CASE(YoloCoOccurrenceLayerTest, TestDtypesAndDevices);

TYPED_TEST(YoloCoOccurrenceLayerTest, TestForward) {
    typedef typename TypeParam::Dtype Dtype;
    this->Initialize({ 0, 3 },
                     { 2, 1 },
                     { 1, 2, 3 },
                     { 0.95f, 0.9f, 1.0f },
                     { 0.1f, 0.6f, 0.6f },
                     { 0.25f, 0.2f, 0.2f },
                     CMAKE_SOURCE_DIR "caffe/test/test_data/yoloco_labelmap_5c.txt",
                     CMAKE_SOURCE_DIR "caffe/test/test_data/yoloco_comap_5c_3.txt");

    LayerParameter layer_param;
    layer_param.mutable_yoloco_param()->set_labelmap(this->labelmap_file_name_);
    layer_param.mutable_yoloco_param()->set_comap(this->comap_file_name_);
    scoped_ptr<YoloCoOccurrenceLayer<Dtype>> layer(new YoloCoOccurrenceLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    this->TestForward();
}


}  // namespace caffe
