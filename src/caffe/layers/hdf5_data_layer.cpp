/*
TODO:
- load file in a separate thread ("prefetch")
- can be smarter about the memcpy call instead of doing it row-by-row
  :: use util functions caffe_copy, and Blob->offset()
  :: don't forget to update hdf5_daa_layer.cu accordingly
- add ability to shuffle filenames if flag is set
*/
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "hdf5.h"
#include "hdf5_hl.h"
#include "stdint.h"

#include "caffe/layers/hdf5_data_layer.hpp"
#include "caffe/util/hdf5.hpp"

#ifdef USE_MPI
#include "caffe/clusters.hpp"
#endif

namespace caffe {

template <typename Dtype>
HDF5DataLayer<Dtype>::~HDF5DataLayer<Dtype>() { }

// Load data and label from HDF5 filename into the class property blobs.
template <typename Dtype>
void HDF5DataLayer<Dtype>::LoadHDF5FileData(const char* filename) {
  DLOG(INFO) << "Loading HDF5 file: " << filename;
  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    LOG(FATAL) << "Failed opening HDF5 file: " << filename;
  }

  int top_size = this->layer_param_.top_size();
  hdf_blobs_.resize(top_size);

  const int MIN_DATA_DIM = 1;
  const int MAX_DATA_DIM = INT_MAX;

  for (int i = 0; i < top_size; ++i) {
    hdf_blobs_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
    // Allow reshape here, as we are loading data not params
    hdf5_load_nd_dataset(file_id, this->layer_param_.top(i).c_str(),
        MIN_DATA_DIM, MAX_DATA_DIM, hdf_blobs_[i].get(), true);
  }

  herr_t status = H5Fclose(file_id);
  CHECK_GE(status, 0) << "Failed to close HDF5 file: " << filename;

  // MinTopBlobs==1 guarantees at least one top blob
  CHECK_GE(hdf_blobs_[0]->num_axes(), 1) << "Input must have at least 1 axis.";
  const int num = hdf_blobs_[0]->shape(0);
  for (int i = 1; i < top_size; ++i) {
    CHECK_EQ(hdf_blobs_[i]->shape(0), num);
  }
  // Default to identity permutation.
  data_permutation_.clear();
  data_permutation_.resize(hdf_blobs_[0]->shape(0));
  for (int i = 0; i < hdf_blobs_[0]->shape(0); i++)
    data_permutation_[i] = i;

  // Shuffle if needed.
  if (this->layer_param_.hdf5_data_param().shuffle()) {
    std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
    DLOG(INFO) << "Successfully loaded " << hdf_blobs_[0]->shape(0)
               << " rows (shuffled)";
  } else {
    DLOG(INFO) << "Successfully loaded " << hdf_blobs_[0]->shape(0) << " rows";
  }
}

template <typename Dtype>
void HDF5DataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  const HDF5DataParameter &hdf5_param = this->layer_param().hdf5_data_param();
  CHECK_EQ(hdf5_param.has_world_size(), hdf5_param.has_world_rank())
      << "world_size and world_rank must be specified together";
  world_size_ = hdf5_param.world_size();
  world_rank_ = hdf5_param.world_rank();
#ifdef USE_MPI
  if (!hdf5_param.has_world_size()) {
    world_size_ = Clusters::node_count();
    world_rank_ = Clusters::node_rank();
  }
#endif
  CHECK_LT(world_rank_, world_size_);
  CHECK_GT(world_size_, 0);

    // Refuse transformation parameters since HDF5 is totally generic.
  CHECK(!this->layer_param_.has_transform_param()) <<
      this->type() << " does not transform data.";
  // Read the source to parse the filenames.
  const string& source = this->layer_param_.hdf5_data_param().source();
  LOG(INFO) << "Loading list of HDF5 filenames from: " << source;
  hdf_filenames_.clear();
  std::ifstream source_file(source.c_str());
  if (source_file.is_open()) {
    std::string line;
    while (source_file >> line) {
      hdf_filenames_.push_back(line);
    }
  } else {
    LOG(FATAL) << "Failed to open source file: " << source;
  }
  source_file.close();
  num_files_ = hdf_filenames_.size();
  current_file_ = 0;
  LOG(INFO) << "Number of HDF5 files: " << num_files_;
  CHECK_GE(num_files_, 1) << "Must have at least 1 HDF5 filename listed in "
    << source;

  file_permutation_.clear();
  file_permutation_.resize(num_files_);
  // Default to identity permutation.
  for (int i = 0; i < num_files_; i++) {
    file_permutation_[i] = i;
  }

  // Shuffle if needed.
  if (this->layer_param_.hdf5_data_param().shuffle()) {
    std::random_shuffle(file_permutation_.begin(), file_permutation_.end());
  }

  // Load the first HDF5 file and initialize the line counter.
  LoadHDF5FileData(hdf_filenames_[file_permutation_[current_file_]].c_str());
  current_row_ = 0;

  // Reshape blobs.
  const int batch_size = this->layer_param_.hdf5_data_param().batch_size();
  const int top_size = this->layer_param_.top_size();
  vector<int> top_shape;
  for (int i = 0; i < top_size; ++i) {
    top_shape.resize(hdf_blobs_[i]->num_axes());
    top_shape[0] = batch_size;
    for (int j = 1; j < top_shape.size(); ++j) {
      top_shape[j] = hdf_blobs_[i]->shape(j);
    }
    top[i]->Reshape(top_shape);
  }
}

template <typename Dtype>
bool HDF5DataLayer<Dtype>::Skip() {
  // In test mode, only rank 0 (of each node) runs, so avoid skipping within a node
  if (this->layer_param_.phase() == TEST)
      return offset_ % world_size_ != 0;  // for world_size == 1, this is always false
  int size = Caffe::solver_count() * world_size_;
  int rank = Caffe::solver_count() * world_rank_ + Caffe::solver_rank();
  bool keep = (offset_ % size) == rank;
  return !keep;
}

template<typename Dtype>
void HDF5DataLayer<Dtype>::Next() {
  if (++current_row_ == hdf_blobs_[0]->shape(0)) {
    if (num_files_ > 1) {
      ++current_file_;
      if (current_file_ == num_files_) {
        current_file_ = 0;
        if (this->layer_param_.hdf5_data_param().shuffle()) {
          std::random_shuffle(file_permutation_.begin(),
                              file_permutation_.end());
        }
        DLOG(INFO) << "Looping around to first file.";
      }
      LoadHDF5FileData(
        hdf_filenames_[file_permutation_[current_file_]].c_str());
    }
    current_row_ = 0;
    if (this->layer_param_.hdf5_data_param().shuffle())
      std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
  }
  offset_++;
}

template <typename Dtype>
void HDF5DataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.hdf5_data_param().batch_size();
  for (int i = 0; i < batch_size; ++i) {
    while (Skip()) {
      Next();
    }
    for (int j = 0; j < this->layer_param_.top_size(); ++j) {
      int data_dim = top[j]->count() / top[j]->shape(0);
      caffe_copy(data_dim,
          &hdf_blobs_[j]->cpu_data()[data_permutation_[current_row_]
            * data_dim], &top[j]->mutable_cpu_data()[i * data_dim]);
    }
    Next();
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(HDF5DataLayer, Forward);
#endif

INSTANTIATE_CLASS(HDF5DataLayer);
REGISTER_LAYER_CLASS(HDF5Data);

}  // namespace caffe
