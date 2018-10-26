#include <vector>
#include <cctype>

#include <stdlib.h>
#include <float.h>
#include "caffe/layers/yolo_co_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/region_common.hpp"

namespace {

// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(),
                         std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
}

char *fgetl(FILE *fp) {
    if (feof(fp))
        return 0;
    size_t size = 512;
    char *line = (char *)malloc(size * sizeof(char));
    if (!fgets(line, size, fp)) {
        free(line);
        return 0;
    }

    size_t curr = strlen(line);

    while ((line[curr - 1] != '\n' || line[curr - 1] != '\r') && !feof(fp)) {
        if (curr == size - 1) {
            size *= 2;
            line = (char *)realloc(line, size * sizeof(char));
            CHECK(line) << size << " could not be allocated";
        }
        size_t readsize = size - curr;
        if (readsize > INT_MAX)
            readsize = INT_MAX - 1;
        fgets(&line[curr], readsize, fp);
        curr = strlen(line);
    }
    if (line[curr - 1] == '\n' || line[curr - 1] == '\r')
        line[curr - 1] = '\0';

    return line;
}

}

namespace caffe {

template <typename Dtype>
void YoloCoOccurrenceLayer<Dtype>::load_labelmap(const string &filename) {
    std::ifstream labelmap_file;
    labelmap_file.open(filename.c_str());
    CHECK(!labelmap_file.fail()) << "Failed to open labelmap file: " << filename << ", error: " << strerror(errno);

    int id = 0;
    labelmap_.clear();
    while (!labelmap_file.eof()) {
        std::string line;
        std::getline(labelmap_file, line);
        rtrim(line);
        if (line.length() == 0) {
            LOG(WARNING) << "labelmap: " << filename << " ended at line: " << id;
            break;
        }

        labelmap_[line] = id++;
    }

    labelmap_file.close();
}

template <typename Dtype>
void YoloCoOccurrenceLayer<Dtype>::load_comap(const string &filename) {
    const YoloCoParameter& yoloco_param = this->layer_param().yoloco_param();
    float default_thresh = yoloco_param.thresh();
    float default_obj_thresh = yoloco_param.obj_thresh();
    float default_ix_thresh = yoloco_param.ix_thresh();

    FILE *fp = fopen(filename.c_str(), "r");
    CHECK(fp) << "Cannot open the co-occurrence map file: " << filename;
    comap_class_cpu_ptr_ = NULL;
    comap_offset_cpu_ptr_ = NULL;
    comap_size_cpu_ptr_ = NULL;
    comap_cpu_ptr_ = NULL;
    comap_thresh_cpu_ptr_ = NULL;
    comap_obj_thresh_cpu_ptr_ = NULL;
    comap_ixr_cpu_ptr_ = NULL;

    char part[256];
    char whole[256];

    int last_c = -1;
    int offset = 0;
    int cidx = 0;
    int co_classes = 0;
    char *line;
    int n = 0;
    while ((line = fgetl(fp)) != 0) {
        float ix_thresh = default_ix_thresh;
        float thresh = default_thresh;
        float obj_thresh = default_obj_thresh;
        int count = sscanf(line, "%[^\t]\t%[^\t]\t%f %f %f", part, whole, &ix_thresh, &thresh, &obj_thresh);
        CHECK_GE(count, 2) << "Few arguments reading line: " << n << " in file:" << filename;
        CHECK_LE(count, 5) << "Error reading line: " << n << " in file:" << filename;

        auto it = labelmap_.find(part);
        CHECK(it != labelmap_.end()) << " line: " << n << " " << part << " is not in labelmap";
        int c = it->second;
        if (c != last_c) {
            last_c = c;
            // Append new class
            for (cidx = 0; cidx < co_classes; ++cidx) {
                if (comap_class_cpu_ptr_[cidx] == c)
                    break;
            }
            CHECK_EQ(cidx, co_classes) << " line: " << n << "class: '" << part << "' co-occurrence group discontinuity";
            co_classes++;
            offset = n;
            comap_class_cpu_ptr_ = (int *)realloc(comap_class_cpu_ptr_, co_classes * sizeof(int));
            comap_class_cpu_ptr_[cidx] = c;
            comap_offset_cpu_ptr_ = (int *)realloc(comap_offset_cpu_ptr_, co_classes * sizeof(int));
            comap_offset_cpu_ptr_[cidx] = offset;
            comap_size_cpu_ptr_ = (int *)realloc(comap_size_cpu_ptr_, co_classes * sizeof(int));
            comap_size_cpu_ptr_[cidx] = 0;
        }
        it = labelmap_.find(whole);
        CHECK(it != labelmap_.end()) << " line: " << n << " '" << whole << "' is not in labelmap";
        int co = it->second;

        int size = comap_size_cpu_ptr_[cidx];
        for (int coidx = 0; coidx < size; ++coidx) {
            CHECK_NE(comap_cpu_ptr_[coidx + offset], co)
                << " line: " << n << "class: " << part << "coclass: " << whole << " earlier line: " << coidx + offset;
        }
        comap_cpu_ptr_ = (int *)realloc(comap_cpu_ptr_, (n + 1) * sizeof(int));
        comap_cpu_ptr_[n] = co;
        comap_ixr_cpu_ptr_ = (float *)realloc(comap_ixr_cpu_ptr_, (n + 1) * sizeof(float));
        comap_ixr_cpu_ptr_[n] = ix_thresh;
        comap_thresh_cpu_ptr_ = (float *)realloc(comap_thresh_cpu_ptr_, (n + 1) * sizeof(float));
        comap_thresh_cpu_ptr_[n] = thresh;
        comap_obj_thresh_cpu_ptr_ = (float *)realloc(comap_obj_thresh_cpu_ptr_, (n + 1) * sizeof(float));
        comap_obj_thresh_cpu_ptr_[n] = obj_thresh;

        comap_size_cpu_ptr_[cidx] = size + 1;

        ++n;
    }
    fclose(fp);

    comap_class_.Reshape({ co_classes });
    comap_class_.set_cpu_data(comap_class_cpu_ptr_);
    comap_offset_.Reshape({ co_classes });
    comap_offset_.set_cpu_data(comap_offset_cpu_ptr_);
    comap_size_.Reshape({ co_classes });
    comap_size_.set_cpu_data(comap_size_cpu_ptr_);

    comap_.Reshape({ n });
    comap_.set_cpu_data(comap_cpu_ptr_);
    comap_thresh_.Reshape({ n });
    comap_thresh_.set_cpu_data(comap_thresh_cpu_ptr_);
    comap_obj_thresh_.Reshape({ n });
    comap_obj_thresh_.set_cpu_data(comap_obj_thresh_cpu_ptr_);
    comap_ixr_.Reshape({ n });
    comap_ixr_.set_cpu_data(comap_ixr_cpu_ptr_);
}

template <typename Dtype>
void YoloCoOccurrenceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    const YoloCoParameter& yoloco_param = this->layer_param().yoloco_param();
    CHECK(yoloco_param.has_labelmap()) << "labelmap file is expected";
    CHECK(yoloco_param.has_comap()) << "CoOccurrence map file is expected";
    // Parse labelmap file and CoOccurrence map file
    load_labelmap(yoloco_param.labelmap());
    CHECK(!labelmap_.empty()) << "labelmap file: '" << yoloco_param.labelmap() << "' is empty";
    load_comap(yoloco_param.comap());
}

template <typename Dtype>
void YoloCoOccurrenceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    Layer<Dtype>::LayerSetUp(bottom, top);

    int blob_idx = 0;
    auto blob_obj = bottom[blob_idx++];
    auto blob_no_obj = bottom[blob_idx++];
    auto blob_truth = bottom[blob_idx++];
    auto blob_bbs = bottom[blob_idx++];
    auto blob_pred = bottom[blob_idx++];

    CHECK_GE(blob_bbs->num_axes(), 3);
    int bbs_axis = blob_bbs->num_axes() - 1;
    CHECK_EQ(blob_bbs->shape(bbs_axis), 4);
    outer_num_ = blob_bbs->shape(0);
    inner_num_ = blob_bbs->count(1, bbs_axis);
    CHECK_EQ(blob_bbs->count(), outer_num_ * inner_num_ * 4);

    CHECK_EQ(blob_obj->num(), outer_num_);
    CHECK_EQ(blob_no_obj->num(), outer_num_);
    CHECK_EQ(blob_truth->num(), outer_num_);
    CHECK_EQ(blob_pred->shape(0), outer_num_);

    CHECK_EQ(blob_obj->count(1), inner_num_);
    CHECK_EQ(blob_no_obj->count(1), inner_num_);

    max_gt_ = blob_truth->channels() / 5;
    CHECK_EQ(5 * max_gt_, blob_truth->channels());
    CHECK_EQ(blob_truth->height(), 1);
    CHECK_EQ(blob_truth->width(), 1);

    CHECK_GE(blob_pred->num_axes(), 4);
    channels_ = blob_pred->shape(1);
    classes_ = labelmap_.size();
    CHECK(channels_ == classes_ || channels_ == classes_ + 1) <<
        "Channel count: " << channels_ << " must match label count = " << classes_ << " [ + 1]";
    CHECK_EQ(blob_pred->count(2), inner_num_);

    auto target_no_obj = top[0];
    target_no_obj->ReshapeLike(*blob_no_obj);
}

template <typename Dtype>
void YoloCoOccurrenceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    int blob_idx = 0;
    auto blob_obj = bottom[blob_idx++];
    auto blob_no_obj = bottom[blob_idx++];
    auto blob_truth = bottom[blob_idx++];
    auto blob_bbs = bottom[blob_idx++];
    auto blob_pred = bottom[blob_idx++];

    auto target_no_obj = top[0];

    caffe_copy(blob_no_obj->count(), blob_no_obj->cpu_data(), target_no_obj->mutable_cpu_data());

    auto co_classes = comap_class_.count();
    if (!co_classes)
        return;

    auto comap_class_data = comap_class_.cpu_data();
    auto comap_offset_data = comap_offset_.cpu_data();
    auto comap_size_data = comap_size_.cpu_data();
    auto comap_data = comap_.cpu_data();
    auto comap_thresh_data = comap_thresh_.cpu_data();
    auto comap_obj_thresh_data = comap_obj_thresh_.cpu_data();
    auto comap_ixr_data = comap_ixr_.cpu_data();
    auto pred_data = blob_pred->cpu_data();
    auto bbs_data = blob_bbs->cpu_data();
    auto truth_data = blob_truth->cpu_data();
    auto obj_data = blob_obj->cpu_data();

    auto target_no_obj_data = target_no_obj->mutable_cpu_data();
    bool with_objectness = (channels_ == classes_ + 1);

#pragma omp parallel for
    for (int index = 0; index < outer_num_ * inner_num_; ++index) {
        // index == n * inner_num_ + s
        const int n = index / inner_num_;
        const int s = index % inner_num_;
        auto obj_index = index;

        // If this is a ground-truth already, nothing to do
        if (target_no_obj_data[obj_index] > 0)
            continue;

        int bbs_index = obj_index * 4;
        Dtype px = *(bbs_data + bbs_index + 0);
        Dtype py = *(bbs_data + bbs_index + 1);
        Dtype pw = *(bbs_data + bbs_index + 2);
        Dtype ph = *(bbs_data + bbs_index + 3);
        // Same as ground-truth logic:
        // we explicitly ignore this zero-length bounding boxes
        if (pw <= 0.00001 || ph <= 0.00001)
            continue;

        auto offset_pred = n * channels_ * inner_num_ + s;
        Dtype objectness;
        if (with_objectness)
            objectness = pred_data[offset_pred + classes_ * inner_num_];
        else
            objectness = obj_data[obj_index];
        bool found = false;
        for (int cidx = 0; cidx < co_classes && !found; ++cidx) {
            auto size = comap_size_data[cidx];
            auto offset = comap_offset_data[cidx];
            auto c = comap_class_data[cidx];
            auto conf = pred_data[offset_pred + c * inner_num_];

            for (int t = 0; t < max_gt_ && !found; ++t) {
                auto offset_nt = n * 5 * max_gt_ + t * 5;
                Dtype tx = *(truth_data + offset_nt + 0);
                // If no ground-truth at this index
                if (!tx)
                    break;
                Dtype ty = *(truth_data + offset_nt + 1);
                Dtype tw = *(truth_data + offset_nt + 2);
                Dtype th = *(truth_data + offset_nt + 3);
                int cls = *(truth_data + offset_nt + 4); // Ground-truth class
                // we explicitly ignore this zero-length bounding boxes
                if (tw <= 0.00001 || th <= 0.00001)
                    continue;

                for (int i = 0; i < size; ++i) {
                    auto co = comap_data[offset + i]; // class that c may co-occur with
                    if (co != cls)
                        continue;
                    // c may co-occure with co only in one rule, so after this the loop will end

                    auto obj_thresh = comap_obj_thresh_data[offset + i];
                    if (objectness < obj_thresh)
                        break;

                    auto thresh = comap_thresh_data[offset + i];
                    if (conf < thresh)
                        break;
                    // Check intersection with co-occured class
                    auto ixr_thresh = comap_ixr_data[offset + i];
                    auto ix = TBoxIntersection(px, py, pw, ph,
                                               tx, ty, tw, th);
                    ix /= (pw * ph); // intersection ratio
                    if (ix >= ixr_thresh) {
                        target_no_obj_data[obj_index] = obj_data[obj_index];
                        found = true;
                    }

                    break;
                }
            }
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(YoloCoOccurrenceLayer);
#endif

INSTANTIATE_CLASS(YoloCoOccurrenceLayer);
REGISTER_LAYER_CLASS(YoloCoOccurrence);

}  // namespace caffe
