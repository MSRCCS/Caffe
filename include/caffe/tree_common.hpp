#ifndef CAFFE_TREE_COMMON_HPP_
#define CAFFE_TREE_COMMON_HPP_

#include <vector>
#include "caffe/blob.hpp"

namespace caffe {

class Tree {
private:
    int* leaf_;
    int n_; // Total number of nodes in the tree
    int* parent_cpu_ptr_; // Parent node of a node
    int* child_cpu_ptr_;  // Initial child group of a node
    int* child_size_cpu_ptr_;  // Child sub-group count of a node
    int* group_cpu_ptr_;  // Group of a node
    char** name_;

    int groups_; // Number of groups in the tree
    int* group_size_cpu_ptr_;
    int* group_offset_cpu_ptr_;

public:
    Tree() : leaf_(NULL), parent_cpu_ptr_(NULL), child_cpu_ptr_(NULL), child_size_cpu_ptr_(NULL),
        group_cpu_ptr_(NULL), name_(NULL), groups_(0),
        group_size_cpu_ptr_(NULL), group_offset_cpu_ptr_(NULL),
        group_size_(), group_offset_(), parent_(), group_(), child_(), child_size_() {
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
    Blob<int> child_;
    Blob<int> child_size_;
};

void read_map(const char *filename, int max_label, Blob<int>& label_map);

}
#endif  // CAFFE_EUCLIDEAN_LOSS_LAYER_HPP_
