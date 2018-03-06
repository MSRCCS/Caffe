#ifndef CAFFE_SOFTMAXTREEPREDICTION_COMMON_HPP_
#define CAFFE_SOFTMAXTREEPREDICTION_COMMON_HPP_

namespace caffe {

template <typename Dtype>
struct TPredictTreeData {
    int outer_num_;
    int channels_;
    int inner_num_;
    bool append_max_;
    float threshold_;
    const int* group_offset_data_; 
    const int* group_size_data_;
    const int* child_data_;
    const int* child_size_data_;
    const Dtype* obj_data_;
    const Dtype* prob_data_;

    TPredictTreeData::TPredictTreeData(int outer_num, int channels, int inner_num,
                                       bool append_max,
                                       const float threshold,
                                       const int* group_offset_data, const int* group_size_data, const int* child_data, const int* child_size_data,
                                       const Dtype* obj_data, const Dtype* prob_data)
    : outer_num_(outer_num), channels_(channels), inner_num_(inner_num)
    , append_max_(append_max)
    , threshold_(threshold)
    , group_offset_data_(group_offset_data), group_size_data_(group_size_data), child_data_(child_data), child_size_data_(child_size_data)
    , obj_data_(obj_data), prob_data_(prob_data) {
    }
};

template <typename Dtype>
__host__ __device__ bool predict_tree(const TPredictTreeData<Dtype>& tpd,
                                      int n, int s, int g, 
                                      Dtype* top_data, double p = 1) {
    int argmax = 0;
    {
        Dtype maxval = -FLT_MAX;
        auto offset = tpd.group_offset_data_[g];
        auto size = tpd.group_size_data_[g];
        for (int j = 0; j < size; ++j) {
            Dtype prob = tpd.prob_data_[(n * tpd.channels_ + offset + j) * tpd.inner_num_ + s];
            if (prob > maxval) {
                argmax = offset + j;
                maxval = prob;
            }
        }
        p *= maxval;
    }
    if (p <= tpd.threshold_)
        return false;

    g = tpd.child_data_[argmax];
    if (g >= 0) {
        // Recurse to each subgroup
        int sg_count = tpd.child_size_data_[argmax] + 1;
        bool all_subgroups = true;
        for (int sg = 0; sg < sg_count; ++sg) {
            if (!predict_tree(tpd,
                              n, s, g + sg,
                              top_data, p)) {
                all_subgroups = false;
            }
        }
        // if all the child subgroups pass the threshold, do not set the parent anymore
        if (all_subgroups)
            return true;
    }

    const int top_channels = tpd.append_max_ ? tpd.channels_ + 1 : tpd.channels_;
    Dtype node_p = tpd.obj_data_ ? tpd.obj_data_[n * tpd.inner_num_ + s] : static_cast<Dtype>(p);
    top_data[(n * top_channels + argmax) * tpd.inner_num_ + s] = node_p;
    if (tpd.append_max_) {
        int max_idx = (n * top_channels + tpd.channels_) * tpd.inner_num_ + s;
        if (node_p > top_data[max_idx])
            top_data[max_idx] = node_p;
    }

    return true;
}

}  // namespace caffe

#endif  // CAFFE_SOFTMAXTREEPREDICTION_COMMON_HPP_
