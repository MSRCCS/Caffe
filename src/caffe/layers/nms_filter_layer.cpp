#include <vector>
#include <numeric>

#include <stdlib.h>
#include <float.h>
#include "caffe/layers/nms_filter_layer.hpp"
#include "caffe/region_common.hpp"

namespace caffe {

template <typename Dtype>
void NMSFilterLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const NMSFilterParameter &nmsfilter_param = this->layer_param().nmsfilter_param();
    nms_ = nmsfilter_param.threshold();
    classes_ = nmsfilter_param.classes();
}

template <typename Dtype>
void NMSFilterLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    Layer<Dtype>::LayerSetUp(bottom, top);

    int blob_idx = 0;
    auto blob_bbs = bottom[blob_idx++];
    auto blob_conf = bottom[blob_idx++];

    CHECK_GE(blob_bbs->num_axes(), 3);
    CHECK_GE(blob_conf->num_axes(), 3);
    int bbs_axis = blob_bbs->num_axes() - 1;
    CHECK_EQ(blob_bbs->shape(bbs_axis), 4);

    outer_num_ = blob_bbs->shape(0);
    CHECK_EQ(blob_conf->shape(0), outer_num_);
    inner_num_ = blob_bbs->count(1, bbs_axis);
    CHECK_EQ(blob_bbs->count(), inner_num_ * outer_num_ * 4);
    int spatial_axis = 2;
    if (blob_conf->count(1) == inner_num_) {
        // Implicite M == 1
        spatial_axis = 1;
        channels_ = 1;
    } else {
        channels_ = blob_conf->shape(1);
    }
    CHECK_LE(classes_, channels_);
    CHECK_EQ(blob_conf->count(spatial_axis), inner_num_);
    CHECK_EQ(blob_conf->count(), inner_num_ * channels_ * outer_num_);

    top[0]->ReshapeLike(*blob_conf);

#ifndef CPU_ONLY
    if (Caffe::mode() == Caffe::GPU) {
        idx_.Reshape({ outer_num_, inner_num_ });
        int block_size = sizeof(unsigned int) * 8;
        int count = outer_num_ * inner_num_ * (inner_num_ - 1) / 2;
        mask_.Reshape({ (count + block_size - 1) / block_size });
    }
#endif
}

// sort the values in p in descending order and keep the index in result
template <typename Dtype>
void sort_nms_idx(const Dtype* p,
                  int inner_num,
                  int c,
                  vector<int>& result) {
    std::iota(result.begin(), result.end(), 0);
    std::sort(result.begin(), result.end(),
              [p, inner_num, c](int i, int j) {
        return p[c * inner_num + i] > p[c * inner_num + j];
    });
}

template <typename Dtype>
void nms_filter(const Dtype* bbs_data,
                int outer_num, int channels, int inner_num, int classes,
                float thresh,
                Dtype* top_conf_data) {

#pragma omp parallel for
    for (int index = 0; index < outer_num * classes; ++index) {
        int c = index % classes;
        int n = index / classes;

        int dim = n * channels * inner_num;
        vector<int> idx(inner_num);
        sort_nms_idx<Dtype>(top_conf_data + dim,
                            inner_num, 
                            c,
                            idx);

        // TODO: profile the performance and try vectorizing with BLAS
        for (int i = 0; i < inner_num; ++i) {
            if (top_conf_data[dim + c * inner_num + idx[i]] == 0)
                continue;
            auto i_bb = bbs_data + (n * inner_num + idx[i]) * 4;
            for (int j = i + 1; j < inner_num; ++j) {
                auto j_bb = bbs_data + (n * inner_num + idx[j]) * 4;
                Dtype curr_iou = TBoxIou<Dtype>(i_bb[0], i_bb[1], i_bb[2], i_bb[3],
                                                j_bb[0], j_bb[1], j_bb[2], j_bb[3]);
                if (curr_iou > thresh)
                    top_conf_data[dim + c * inner_num + idx[j]] = 0;
            }
        }
    }
}

template <typename Dtype>
void NMSFilterLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    int blob_idx = 0;
    auto blob_bbs = bottom[blob_idx++];
    auto blob_conf = bottom[blob_idx++];

    auto top_conf = top[0];
    auto top_conf_data = top_conf->mutable_cpu_data();
    caffe_copy(blob_conf->count(), blob_conf->cpu_data(), top_conf_data);

    auto bbs_data = blob_bbs->cpu_data();

    if (nms_ <= 0 || inner_num_ == 1)
        return;

    int actual_classes = classes_;
    if (actual_classes <= 0)
        actual_classes = channels_;
    nms_filter(bbs_data,
               outer_num_, channels_, inner_num_, actual_classes,
               nms_,
               top_conf_data);
}

#ifdef CPU_ONLY
STUB_GPU(NMSFilterLayer);
#endif

INSTANTIATE_CLASS(NMSFilterLayer);
REGISTER_LAYER_CLASS(NMSFilter);

}  // namespace caffe
