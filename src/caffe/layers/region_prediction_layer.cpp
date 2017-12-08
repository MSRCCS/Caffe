#include <vector>
#include <numeric>

#include <stdlib.h>
#include <float.h>
#include "caffe/layers/region_prediction_layer.hpp"
#include "caffe/region_common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RegionPredictionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    const RegionPredictionParameter &region_param = this->layer_param().region_prediction_param();
    thresh_ = region_param.thresh();
    nms_ = region_param.nms();
    feat_stride_ = region_param.feat_stride();

    biases_.Reshape(region_param.biases_size(), 1, 1, 1);
    for (int i = 0; i < region_param.biases_size(); ++i) {
        *(biases_.mutable_cpu_data() + i) = region_param.biases(i);
    }

    CHECK(biases_.count() % 2 == 0) << "the number of biases must be even: " << biases_.count();
    class_specific_nms_ = region_param.class_specific_nms();
}

template <typename Dtype>
void RegionPredictionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    Layer<Dtype>::LayerSetUp(bottom, top);

    int blob_idx = 0;
    auto blob_xy = bottom[blob_idx++];
    blob_idx++;
    blob_idx++;
    auto blob_conf = bottom[blob_idx++];

    blob_idx = 0;
    auto bbs = top[blob_idx++];
    auto prob = top[blob_idx++];
    
    int num = blob_xy->num();
    int num_anchor = blob_xy->channels() / 2;
    int height = blob_xy->height();
    int width = blob_xy->width();
    int classes = blob_conf->channels();
    CHECK_EQ(blob_conf->num(), num_anchor * num);
    CHECK_EQ(blob_conf->height(), height);
    CHECK_EQ(blob_conf->width(), width);
    CHECK_EQ(biases_.count(), num_anchor * 2);

    vector<int> shape = {num, num_anchor, height, width, 4};
    bbs->Reshape(shape);
    shape[shape.size() - 1] = classes + 1; 
    prob->Reshape(shape);
}

template <typename Dtype>
void correct_region_boxes(Blob<Dtype> *bbs, int im_w, int im_h, int netw, int neth)
{
    int new_w = 0;
    int new_h = 0;
    if (((Dtype)netw / im_w) < ((Dtype)neth / im_h)) {
        new_w = netw;
        new_h = (im_h * netw) / im_w;
    }
    else {
        new_h = neth;
        new_w = (im_w * neth) / im_h;
    }
    auto bbs_data = bbs->mutable_cpu_data();
    int n = bbs->count() / 4;
    for (int i = 0; i < n; ++i) {
        Dtype x = bbs_data[4 * i + 0];
        Dtype y = bbs_data[4 * i + 1];
        Dtype w = bbs_data[4 * i + 2];
        Dtype h = bbs_data[4 * i + 3];

        x = (x - (netw - new_w) / 2. / netw) / ((Dtype)new_w / netw);
        y = (y - (neth - new_h) / 2. / neth) / ((Dtype)new_h / neth);
        w *= (Dtype)netw / new_w;
        h *= (Dtype)neth / new_h;
        x *= im_w;
        w *= im_w;
        y *= im_h;
        h *= im_h;
        bbs_data[4 * i + 0] = x;
        bbs_data[4 * i + 1] = y;
        bbs_data[4 * i + 2] = w;
        bbs_data[4 * i + 3] = h;
    }
}

// sort the values in p in descending order and keep the index in result
template <typename Dtype>
void sort_idx(const Dtype* p, int n, int stride, vector<int>& result) {
    result.resize(n);
    std::iota(result.begin(), result.end(), 0);
    std::sort(result.begin(), result.end(),
            [p, stride](int i1, int i2) {return p[i1 * stride] > p[i2 * stride];});
}

template <typename Dtype>
void do_nms_sort(Blob<Dtype>* bbs, Blob<Dtype>* prob, Dtype thresh)
{
    const vector<int>& shape = prob->shape();
    int classes = shape[shape.size() - 1] - 1;
    int num_bb = bbs->count() / 4;
    int prob_stride = classes + 1;
    
    auto bbs_data = bbs->cpu_data();
    for (int k = 0; k < classes; ++k) {
        auto p = prob->mutable_cpu_data() + prob->offset({0, 0, 0, 0, k});
        vector<int> idx;
        sort_idx<Dtype>(p, num_bb, prob_stride, idx);

        for (int i = 0; i < num_bb; ++i) {
            if (p[idx[i] * prob_stride] == 0) {
                continue;
            }
            auto i_bb = bbs_data + idx[i] * 4;
            for (int j = i + 1; j < num_bb; ++j) {
                auto j_bb = bbs_data + idx[j] * 4;
                Dtype curr_iou = TBoxIou<Dtype>(i_bb[0], i_bb[1], i_bb[2], i_bb[3], 
                        j_bb[0], j_bb[1], j_bb[2], j_bb[3]);
                if (curr_iou > thresh) {
                    p[idx[j] * prob_stride] = 0;
                }
            }
        }
    }
}

template <typename Dtype>
void do_nms_obj(Blob<Dtype>* bbs, Blob<Dtype>* prob, Dtype thresh)
{
    const vector<int>& shape = prob->shape();
    int classes = shape[shape.size() - 1] - 1;
    int num_bb = bbs->count() / 4;
    int prob_stride = classes + 1;

    auto p = prob->mutable_cpu_data() + prob->offset({0, 0, 0, 0, classes});
    vector<int> idx;
    sort_idx<Dtype>(p, num_bb, prob_stride, idx);
    
    auto bbs_data = bbs->cpu_data();
    for (int i = 0; i < num_bb; ++i) {
        if (p[idx[i] * prob_stride] == 0) {
            continue;
        }
        auto i_bb = bbs_data + idx[i] * 4;
        for (int j = i + 1; j < num_bb; ++j) {
            auto j_bb = bbs_data + idx[j] * 4;
            Dtype curr_iou = TBoxIou<Dtype>(i_bb[0], i_bb[1], i_bb[2], i_bb[3], 
                    j_bb[0], j_bb[1], j_bb[2], j_bb[3]);
            if (curr_iou > thresh) {
                p[idx[j] * prob_stride] = 0;
            }
        }
    }
}

template <typename Dtype>
void RegionPredictionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    int blob_idx = 0;
    auto blob_xy = bottom[blob_idx++];
    auto blob_wh = bottom[blob_idx++];
    auto blob_obj = bottom[blob_idx++];
    auto blob_conf = bottom[blob_idx++];
    auto blob_imageinfo = bottom[blob_idx++];
    
    blob_idx = 0;
    auto bbs = top[blob_idx++];
    auto prob = top[blob_idx++];

    const Dtype* im_info = blob_imageinfo->cpu_data();
    int im_w = im_info[1];
    int im_h = im_info[0];
    // when used for Caffe timing, im_w and im_h might be 0 and we need to give them valid values.
    if (im_w == 0)
        im_w = net_w_;
    if (im_h == 0)
        im_h = net_h_;

    int batches = blob_xy->num();
    int height = blob_xy->height();
    int width = blob_xy->width();
    int num_anchor = blob_xy->channels() / 2;
    int classes = blob_conf->channels();

    Dtype *bbs_data = bbs->mutable_cpu_data();
    Dtype *prob_data = prob->mutable_cpu_data();
    caffe_set(prob->count(), (Dtype)0, prob_data);
    auto biases = biases_.cpu_data();

    for (int b = 0; b < batches; b++) {
        for (int n = 0; n < num_anchor; n++) {
            for (int j = 0; j < height; j++) {
                for (int i = 0; i < width; i++) {
                    *(bbs_data + bbs->offset({b, n, j, i, 0})) = 
                        (blob_xy->data_at(b, n, j, i) + i) / width;
                    *(bbs_data + bbs->offset({b, n, j, i, 1})) = 
                        (blob_xy->data_at(b, n + num_anchor, j, i) + j) / height;
                    *(bbs_data + bbs->offset({b, n, j, i, 2})) = 
                        exp(blob_wh->data_at(b, n, j, i)) * biases[2 * n] / width;
                    *(bbs_data + bbs->offset({b, n, j, i, 3})) = 
                        exp(blob_wh->data_at(b, n + num_anchor, j, i)) * biases[2 * n + 1] / height;
                }
            }
        }
    }


    for (int b = 0; b < batches; b++) {
        for (int n = 0; n < num_anchor; n++) {
            for (int j = 0; j < height; j++) {
                for (int i = 0; i < width; i++) {
                    auto scale = blob_obj->data_at(b, n, j, i);
                    Dtype max = 0;
                    for (int c = 0; c < classes; ++c) {
                        auto p = scale * blob_conf->data_at({b * num_anchor + n, c, j, i});
                        if (p <= thresh_)
                            p = 0;
                        *(prob_data + prob->offset({b, n, j, i, c})) = p;
                        if (p > max) {
                            max = p;
                        }
                    }
                    *(prob_data + prob->offset({b, n, j, i, classes})) = max;
                }
            }
        }
    }

    correct_region_boxes<Dtype>(bbs, im_w, im_h, width * feat_stride_, height * feat_stride_);
    
    if (nms_ > 0) {
        if (class_specific_nms_) {
            do_nms_sort(bbs, prob, nms_);
        } else {
            do_nms_obj(bbs, prob, nms_);//0.4);
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(RegionPredictionLayer);
#endif

INSTANTIATE_CLASS(RegionPredictionLayer);
REGISTER_LAYER_CLASS(RegionPrediction);

}  // namespace caffe
