#include <algorithm>
#include <vector>

#include "caffe/layers/softmaxtree_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace {

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

    while ((line[curr - 1] != '\n') && !feof(fp)) {
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
    if (line[curr - 1] == '\n') 
        line[curr - 1] = '\0';

    return line;
}

caffe::Tree read_tree(const char *filename) {
    caffe::Tree t;
    FILE *fp = fopen(filename, "r");
    CHECK(fp) << "Cannot open the tree file: " << filename;

    char *line;
    int last_parent = -1;
    int group_size = 0;
    int groups = 0;
    int n = 0;
    while ((line = fgetl(fp)) != 0) {
        char *id = (char *)calloc(256, sizeof(char));
        int parent = -1;
        sscanf(line, "%s %d", id, &parent);
        t.parent = (int *)realloc(t.parent, (n + 1) * sizeof(int));
        t.parent[n] = parent;

        t.child = (int *)realloc(t.child, (n + 1) * sizeof(int));
        t.child[n] = -1;

        t.name = (char **)realloc(t.name, (n + 1) * sizeof(char *));
        t.name[n] = id;
        if (parent != last_parent) {
            ++groups;
            t.group_offset_cpu_ptr = (int *)realloc(t.group_offset_cpu_ptr, groups * sizeof(int));
            t.group_offset_cpu_ptr[groups - 1] = n - group_size;
            t.group_size_cpu_ptr = (int *)realloc(t.group_size_cpu_ptr, groups * sizeof(int));
            t.group_size_cpu_ptr[groups - 1] = group_size;
            group_size = 0;
            last_parent = parent;
        }
        t.group = (int *)realloc(t.group, (n + 1) * sizeof(int));
        t.group[n] = groups;
        if (parent >= 0) {
            t.child[parent] = groups;
        }
        ++n;
        ++group_size;
    }
    fclose(fp);

    ++groups;
    t.group_offset_cpu_ptr = (int *)realloc(t.group_offset_cpu_ptr, groups * sizeof(int));
    t.group_offset_cpu_ptr[groups - 1] = n - group_size;
    t.group_size_cpu_ptr = (int *)realloc(t.group_size_cpu_ptr, groups * sizeof(int));
    t.group_size_cpu_ptr[groups - 1] = group_size;

    t.n = n;
    t.groups = groups;
    t.leaf = (int *)calloc(n, sizeof(int));
    int i;
    for (i = 0; i < n; ++i) 
        t.leaf[i] = 1;
    for (i = 0; i < n; ++i) 
        if (t.parent[i] >= 0) 
            t.leaf[t.parent[i]] = 0;

    t.group_offset.reset(new caffe::SyncedMemory(groups * sizeof(int)));
    t.group_offset->set_cpu_data(t.group_offset_cpu_ptr);
    t.group_size.reset(new caffe::SyncedMemory(groups * sizeof(int)));
    t.group_size->set_cpu_data(t.group_size_cpu_ptr);

    return t;
}

}

namespace caffe {

template <typename Dtype>
void SoftmaxTreeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    Layer<Dtype>::LayerSetUp(bottom, top);
    const SoftmaxTreeParameter &softmaxtree_param = this->layer_param().softmaxtree_param();
    softmax_tree_ = read_tree(softmaxtree_param.tree().c_str());
#ifndef CPU_ONLY
    // Pre-fetch data
    if (Caffe::mode() == Caffe::GPU) {
        softmax_tree_.group_size->mutable_gpu_data();
        softmax_tree_.group_offset->mutable_gpu_data();
    }
#endif
}

template <typename Dtype>
void SoftmaxTreeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmaxtree_param().axis());

  int channels = bottom[0]->shape(softmax_axis_);

  // This may requires a reshape layer to reshape to CxA before softmaxtree
  CHECK(channels == softmax_tree_.n) << "Channel count: " << channels << " must match tree node count: " << softmax_tree_.n;

  top[0]->ReshapeLike(*bottom[0]);
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  if (Caffe::mode() == Caffe::CPU) {
      vector<int> mult_dims(1, channels);
      sum_multiplier_.Reshape(mult_dims);
      caffe_set(sum_multiplier_.count(), Dtype(1), sum_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void SoftmaxTreeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  auto group_offset_data = (const int*)softmax_tree_.group_offset->cpu_data();
  auto group_size_data = (const int*)softmax_tree_.group_size->cpu_data();
  int channels = bottom[0]->shape(softmax_axis_);
  int dim = bottom[0]->count() / outer_num_; // == channels * inner_num_
  caffe_copy(bottom[0]->count(), bottom_data, top_data);

  // We need to subtract the per-group max to avoid numerical issues, compute the exp,
  // and then normalize per-group.
  for (int i = 0; i < outer_num_; ++i) {

#pragma omp parallel for
    for (int g = 0; g < softmax_tree_.groups; ++g) {
        auto offset = group_offset_data[g];
        auto size = group_size_data[g];
        for (int k = 0; k < inner_num_; ++k) {
            Dtype maxval = -FLT_MAX;
            for (int j = 0; j < size; ++j) {
                if (bottom_data[(offset + j) * inner_num_ + k] > maxval)
                    maxval = bottom_data[(offset + j) * inner_num_ + k];
            }
            // Subtract the max
            for (int j = 0; j < size; ++j)
                top_data[(offset + j) * inner_num_ + k] -= maxval;
        }
    }

    // exponentiation
    caffe_exp<Dtype>(dim, top_data, top_data);

    // per-group sum after exp, and divide
#pragma omp parallel for
    for (int g = 0; g < softmax_tree_.groups; ++g) {
        auto offset = group_offset_data[g];
        auto size = group_size_data[g];
        for (int k = 0; k < inner_num_; ++k) {
            auto sum = caffe_cpu_strided_dot(size, sum_multiplier_.cpu_data(), 1, &top_data[offset * inner_num_ + k], inner_num_);
            // divide by sum
            for (int j = 0; j < size; ++j)
                top_data[(offset + j) * inner_num_ + k] /= sum;
        }
    }

    top_data += dim;
    bottom_data += dim;
  }
}

template <typename Dtype>
void SoftmaxTreeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  auto group_offset_data = (const int*)softmax_tree_.group_offset->cpu_data();
  auto group_size_data = (const int*)softmax_tree_.group_size->cpu_data();
  int channels = top[0]->shape(softmax_axis_);
  int dim = top[0]->count() / outer_num_; // == channels * inner_num_
  caffe_copy(top[0]->count(), top_diff, bottom_diff);

  // darknet only does this only:
  //caffe_axpy(bottom[0]->count(), Dtype(1.), top_diff, bottom_diff);

  for (int i = 0; i < outer_num_; ++i) {
    // compute per-group dot(top_diff, top_data) and subtract them from the bottom diff
#pragma omp parallel for
      for (int g = 0; g < softmax_tree_.groups; ++g) {
          auto offset = group_offset_data[g];
          auto size = group_size_data[g];
          for (int k = 0; k < inner_num_; ++k) {
              auto dot = caffe_cpu_strided_dot<Dtype>(size, 
                                                      bottom_diff + i * dim + offset * inner_num_ + k, inner_num_, 
                                                      top_data + i * dim + offset * inner_num_ + k, inner_num_);
              // Subtract the dot
              for (int j = 0; j < size; ++j)
                  bottom_diff[i * dim + (offset + j) * inner_num_ + k] -= dot;
          }
      }
  }
  // elementwise multiplication
  caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxTreeLayer);
#endif

INSTANTIATE_CLASS(SoftmaxTreeLayer);
REGISTER_LAYER_CLASS(SoftmaxTree);

}  // namespace caffe
