#include <vector>
#include <numeric>

#include <stdlib.h>
#include <float.h>
#include "caffe/layers/yolo_bbs_layer.hpp"
#include "caffe/region_common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void YoloBBsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const YoloBBsParameter& yolobbs_param = this->layer_param().yolobbs_param();
    feat_stride_ = yolobbs_param.feat_stride();

    CHECK(yolobbs_param.biases_size() > 0) << "biases cannot be empty ";
    CHECK(yolobbs_param.biases_size() % 2 == 0) << "the number of biases must be even: " << yolobbs_param.biases_size();
    biases_.Reshape(yolobbs_param.biases_size(), 1, 1, 1);
    for (int i = 0; i < yolobbs_param.biases_size(); ++i)
        *(biases_.mutable_cpu_data() + i) = yolobbs_param.biases(i);

    with_objectness_ = bottom.size() > 3;
    if (with_objectness_) {
        CHECK_EQ(bottom.size(), 5);
        CHECK_EQ(top.size(), 2);
    } else {
        CHECK_EQ(bottom.size(), 3);
        CHECK_EQ(top.size(), 1);
    }
}

template <typename Dtype>
void YoloBBsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    Layer<Dtype>::LayerSetUp(bottom, top);

    int blob_idx = 0;
    auto blob_xy = bottom[blob_idx++];
    auto blob_wh = bottom[blob_idx++];
    CHECK_EQ(blob_xy->count(), blob_wh->count());

    int num_anchor = blob_xy->channels() / 2;
    CHECK_EQ(num_anchor * 2, biases_.count());
    CHECK_EQ(num_anchor * 2, blob_xy->channels());
    auto height = blob_xy->height();
    auto width = blob_xy->width();
    CHECK_EQ(blob_wh->height(), height);
    CHECK_EQ(blob_wh->width(), width);
    CHECK_EQ(blob_wh->channels(), 2 * num_anchor);

    auto blob_imageinfo = bottom[blob_idx++];
    CHECK_EQ(blob_imageinfo->count(), 2);

    // BBS output
    top[0]->Reshape({ blob_xy->num(), num_anchor, height, width, 4 });

    const YoloBBsParameter& yolobbs_param = this->layer_param().yolobbs_param();
    if (with_objectness_) {
        auto blob_objectness = bottom[blob_idx++];
        auto blob_conf = bottom[blob_idx++];
        CHECK_EQ(blob_objectness->count(), blob_xy->count() / 2);
        auto channels = blob_conf->count() / blob_objectness->count();
        CHECK_GE(channels * blob_objectness->count(), blob_conf->count());

        top[1]->ReshapeLike(*blob_conf);
    }
}

template <typename Dtype>
void correct_bbs(Blob<Dtype> *bbs, int im_w, int im_h, int netw, int neth) {
    int new_w = 0;
    int new_h = 0;
    if (((Dtype)netw / im_w) < ((Dtype)neth / im_h)) {
        new_w = netw;
        new_h = (im_h * netw) / im_w;
    } else {
        new_h = neth;
        new_w = (im_w * neth) / im_h;
    }
    auto bbs_data = bbs->mutable_cpu_data();
    int total = bbs->count() / 4;

#pragma omp parallel for
    for (int i = 0; i < total; ++i) {
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

template <typename Dtype>
void YoloBBsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    int blob_idx = 0;
    auto blob_xy = bottom[blob_idx++];
    auto blob_wh = bottom[blob_idx++];
    auto blob_imageinfo = bottom[blob_idx++];

    auto bbs = top[0];

    int batches = blob_xy->num();
    int height = blob_xy->height();
    int width = blob_xy->width();
    int num_anchor = blob_xy->channels() / 2;

    Dtype *bbs_data = bbs->mutable_cpu_data();
    auto biases_data = biases_.cpu_data();

#pragma omp parallel for
    for (int b = 0; b < batches; b++) {
        for (int n = 0; n < num_anchor; n++) {
            for (int j = 0; j < height; j++) {
                for (int i = 0; i < width; i++) {
                    *(bbs_data + bbs->offset({ b, n, j, i, 0 })) =
                        (blob_xy->data_at(b, n, j, i) + i) / width;
                    *(bbs_data + bbs->offset({ b, n, j, i, 1 })) =
                        (blob_xy->data_at(b, n + num_anchor, j, i) + j) / height;
                    *(bbs_data + bbs->offset({ b, n, j, i, 2 })) =
                        exp(blob_wh->data_at(b, n, j, i)) * biases_data[2 * n] / width;
                    *(bbs_data + bbs->offset({ b, n, j, i, 3 })) =
                        exp(blob_wh->data_at(b, n + num_anchor, j, i)) * biases_data[2 * n + 1] / height;
                }
            }
        }
    }

    int net_h = feat_stride_ * height;
    int net_w = feat_stride_ * width;

    const Dtype* im_info = blob_imageinfo->cpu_data();
    int im_h = im_info[0];
    int im_w = im_info[1];
    // when used for Caffe timing, im_w and im_h might be 0 and we need to give them valid values.
    if (im_w == 0)
        im_w = net_w;
    if (im_h == 0)
        im_h = net_h;

    correct_bbs<Dtype>(bbs, im_w, im_h, net_w, net_h);

    if (with_objectness_) {
        auto blob_objectness = bottom[blob_idx++];
        auto blob_conf = bottom[blob_idx++];
        auto blob_top_conf = top[1];
        auto thresh = this->layer_param().yolobbs_param().thresh();

        auto count = blob_conf->count();
        auto channels = count / blob_objectness->count();
        if (channels == 1) {
            caffe_mul(count,
                      blob_conf->cpu_data(), 
                      blob_objectness->cpu_data(),
                      blob_top_conf->mutable_cpu_data());
        } else {
            auto outer_num = blob_objectness->count(0, 1);
            auto inner_num = blob_objectness->count(1);
            DCHECK_EQ(count, outer_num * channels * inner_num);
#pragma omp parallel for
            for (int i = 0; i < outer_num; ++i) {
                for (int c = 0; c < channels; ++c) {
                    caffe_mul(inner_num,
                              blob_conf->cpu_data() + (i * channels + c) * inner_num,
                              blob_objectness->cpu_data() + i  * inner_num,
                              blob_top_conf->mutable_cpu_data() + (i * channels + c) * inner_num);
                }
            }
        }
        // To avoid any thresholding, we can set a negative threshold
        if (thresh >= 0) {
            auto blob_top_conf_data = blob_top_conf->mutable_cpu_data();
            for (int i = 0; i < count; ++i) {
                if (blob_top_conf_data[i] <= thresh)
                    blob_top_conf_data[i] = 0;
            }
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(YoloBBsLayer);
#endif

INSTANTIATE_CLASS(YoloBBsLayer);
REGISTER_LAYER_CLASS(YoloBBs);

}  // namespace caffe
