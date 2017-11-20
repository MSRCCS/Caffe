#ifndef CAFFE_TREE_COMMON_HPP_
#define CAFFE_TREE_COMMON_HPP_

#include "caffe/blob.hpp"

namespace caffe {

class Tree {
private:
    int* leaf_;
    int n_; // Total number of nodes in the tree
    int* parent_cpu_ptr_;
    int* child_;
    int* group_cpu_ptr_;
    char** name_;

    int groups_; // Number of groups in the tree
    int* group_size_cpu_ptr_;
    int* group_offset_cpu_ptr_;

public:
    Tree() : leaf_(NULL), parent_cpu_ptr_(NULL), child_(NULL),
        group_cpu_ptr_(NULL), name_(NULL), groups_(0),
        group_size_cpu_ptr_(NULL), group_offset_cpu_ptr_(NULL),
        group_size_(), group_offset_(), parent_(), group_() {
    }
    void read(const char *filename);
    int groups() {
        return groups_;
    }
    int nodes() {
        return n_;
    }

    Blob<int> group_size_;
    Blob<int> group_offset_;
    Blob<int> parent_;
    Blob<int> group_;
};

}
#endif  // CAFFE_EUCLIDEAN_LOSS_LAYER_HPP_
