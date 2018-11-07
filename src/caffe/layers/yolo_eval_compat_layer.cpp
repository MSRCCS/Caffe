#include <vector>
#include <numeric>

#include <stdlib.h>
#include <float.h>
#include "caffe/layers/yolo_eval_compat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void YoloEvalCompatLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const YoloEvalCompatParameter& yoloevalcompat_param = this->layer_param().yoloevalcompat_param();
    move_axis_ = yoloevalcompat_param.move_axis();
    append_max_ = yoloevalcompat_param.append_max();
}

template <typename Dtype>
void YoloEvalCompatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    Layer<Dtype>::LayerSetUp(bottom, top);

    const YoloEvalCompatParameter& yoloevalcompat_param = this->layer_param().yoloevalcompat_param();
    sum_inner_num_ = 0;
    sum_classes_ = 0;
    outer_num_ = bottom[0]->shape(0);
    auto bottom_count = bottom.size();
    for (int i = 0; i < bottom_count; ++i) {
        CHECK_EQ(bottom[i]->shape(0), outer_num_);
        CHECK_GE(bottom[i]->num_axes(), 3);
        auto classes = bottom[i]->shape(1);
        // When not appending max, we assume the last column is already the objectness/max
        if (!append_max_) {
            CHECK_GT(classes, 1)
                << "bottom: " << i << " not enough classes";
            classes--;
        }
        sum_classes_ += classes;
        sum_inner_num_ += bottom[i]->count(2);
    }
    int channels = sum_classes_ + 1;
    if (move_axis_)
        top[0]->Reshape({ outer_num_, sum_inner_num_, channels });
    else
        top[0]->Reshape({ outer_num_, channels, sum_inner_num_ });
}

template <typename Dtype>
void YoloEvalCompatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    auto top_data = top[0]->mutable_cpu_data();
    const int bottom_count = bottom.size();
    if (bottom_count > 1)
        caffe_set(top[0]->count(), Dtype(0), top_data);

    int channels = sum_classes_ + 1;
    int c_offset = 0;
    int s_offset = 0;
    for (int i = 0; i < bottom_count; ++i) {
        auto bottom_data = bottom[i]->cpu_data();
        auto inner_num = bottom[i]->count(2);
        auto classes = bottom[i]->shape(1);
        if (!append_max_) {
#pragma omp parallel for
            for (int index = 0; index < outer_num_ * classes * inner_num; ++index) {
                int s = index % inner_num;
                int c = (index / inner_num) % classes;
                const int n = (index / inner_num) / classes;

                auto p = bottom_data[(n * classes + c) * inner_num + s];

                s += s_offset;
                // concatenate objectness
                if (c == classes - 1)
                    c = sum_classes_;
                else
                    c += c_offset;

                if (move_axis_)
                    top_data[(n * sum_inner_num_ + s) * channels + c] = p;
                else
                    top_data[(n * channels + c) * sum_inner_num_ + s] = p;
            }
            classes--;
        } else {
#pragma omp parallel for
            for (int index = 0; index < outer_num_ * inner_num; ++index) {
                // index == n * sum_inner_num_ + s
                const int n = index / inner_num;
                const int s = index % inner_num;
                auto s2 = s + s_offset;
                Dtype maxval = -FLT_MAX;
                if (i > 0) {
                    if (move_axis_)
                        maxval = top_data[(n * sum_inner_num_ + s2) * channels + sum_classes_];
                    else
                        maxval = top_data[(n * channels + sum_classes_) * sum_inner_num_ + s2];
                }
                for (int c = 0; c < classes; ++c) {
                    auto p = bottom_data[(n * classes + c) * inner_num + s];
                    if (p > maxval)
                        maxval = p;
                    auto c2 = c + c_offset;
                    if (move_axis_)
                        top_data[(n * sum_inner_num_ + s2) * channels + c2] = p;
                    else
                        top_data[(n * channels + c2) * sum_inner_num_ + s2] = p;
                }
                if (move_axis_)
                    top_data[(n * sum_inner_num_ + s2) * channels + sum_classes_] = maxval;
                else
                    top_data[(n * channels + sum_classes_) * sum_inner_num_ + s2] = maxval;
            }
        }
        s_offset += inner_num;
        c_offset += classes;
    }
}

#ifdef CPU_ONLY
STUB_GPU(YoloEvalCompatLayer);
#endif

INSTANTIATE_CLASS(YoloEvalCompatLayer);
REGISTER_LAYER_CLASS(YoloEvalCompat);

}  // namespace caffe
