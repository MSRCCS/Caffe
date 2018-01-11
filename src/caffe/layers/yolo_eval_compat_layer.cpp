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
    classes_ = yoloevalcompat_param.classes();
    CHECK_GT(classes_, 0) << "invalid number of classes";
    threshold_ = yoloevalcompat_param.thresh();
}

template <typename Dtype>
void YoloEvalCompatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    Layer<Dtype>::LayerSetUp(bottom, top);

    auto num_axes = bottom[0]->num_axes();
    CHECK_GE(num_axes, 4);
    outer_num_ = bottom[0]->shape(0);
    inner_num_ = bottom[0]->count(num_axes - 3);
    if (bottom.size() == 1) {
        CHECK_EQ(bottom[0]->count(), outer_num_ * classes_ * inner_num_)
            << "With no bottom indices, bottom[0] must have '" << classes_ << "' classes";
    } else {
        CHECK_EQ(bottom[0]->count(), bottom[1]->count()) << "There must be exactly one index for each probability in bottom[0]";
        CHECK_EQ(bottom[0]->count(), outer_num_ * inner_num_);
        CHECK_EQ(outer_num_, bottom[1]->shape(0));
    }
    int num_anchor = bottom[0]->shape(num_axes - 3);
    int height = bottom[0]->shape(num_axes - 2);
    int width = bottom[0]->shape(num_axes - 1);

    top[0]->Reshape({ outer_num_, num_anchor, height, width, classes_ + 1 });
}

template <typename Dtype>
void YoloEvalCompatLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    auto bottom_data = bottom[0]->cpu_data();
    auto top_data = top[0]->mutable_cpu_data();
    if (bottom.size() == 1) {
#pragma omp parallel for
        for (int index = 0; index < outer_num_ * inner_num_; ++index) {
            // index == n * inner_num_ + s
            const int n = index / inner_num_;
            const int s = index % inner_num_;
            Dtype maxval = -FLT_MAX;
            for (int c = 0; c < classes_; ++c) {
                auto p = bottom_data[(n * classes_ + c) * inner_num_ + s];
                if (p <= threshold_)
                    p = 0;
                if (p > maxval)
                    maxval = p;
                top_data[(n * inner_num_ + s) * (classes_ + 1) + c] = p;
            }
            top_data[(n * inner_num_ + s) * (classes_ + 1) + classes_] = maxval;
        }
        return;
    }

    auto class_data = bottom[1]->cpu_data();
    int count = bottom[0]->count();
    caffe_set(count, Dtype(0), top_data);

#pragma omp parallel for
    for (int index = 0; index < count; ++index) {
        auto p = bottom_data[index];
        if (p <= threshold_)
            p = 0;
        int c = (int)class_data[index];
        DCHECK_GE(c, 0);
        DCHECK_LT(c, classes_);
        top_data[index * (classes_ + 1) + c] = p;
        top_data[index * (classes_ + 1) + classes_] = p;
    }
}

#ifdef CPU_ONLY
STUB_GPU(YoloEvalCompatLayer);
#endif

INSTANTIATE_CLASS(YoloEvalCompatLayer);
REGISTER_LAYER_CLASS(YoloEvalCompat);

}  // namespace caffe
