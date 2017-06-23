#ifndef CAFFE_CPM_DATA_TRANSFORMER_HPP
#define CAFFE_CPM_DATA_TRANSFORMER_HPP

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
//using namespace cv;

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/tsv_data_io.hpp" // for base64_decode

namespace caffe {

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */
template <typename Dtype>
class CPMDataTransformer {
 public:
  explicit CPMDataTransformer(const CPMTransformationParameter& param, Phase phase);
  virtual ~CPMDataTransformer() {}

  /**
   * @brief Initialize the Random number generations if needed by the
   *    transformation.
   */
  void InitRand();
  
  void Transform_nv2(const std::string &input_b64coded_data, const std::string &input_label_data, Dtype* transformed_data, Dtype* transformed_label);

  struct AugmentSelection {
    bool flip;
    float degree;
    cv::Size crop;
    float scale;
  };

  struct Joints {
    vector<cv::Point2f> joints;
    vector<float> isVisible;
  };

  struct MetaData {
    std::string dataset;
    cv::Size img_size;
    bool isValidation;
    int numOtherPeople;
    int people_index;
    int annolist_index;
    int write_number;
    int total_write_number;
    int epoch;
    cv::Point2f objpos; //objpos_x(float), objpos_y (float)
    float scale_self;
    Joints joint_self; //(3*16)

    vector<cv::Point2f> objpos_other; //length is numOtherPeople
    vector<float> scale_other; //length is numOtherPeople
    vector<Joints> joint_others; //length is numOtherPeople

	cv::Mat mask_miss;
  };

  void generateLabelMap(Dtype*, cv::Mat&, MetaData meta);

  bool augmentation_flip(cv::Mat& img, cv::Mat& img_aug, cv::Mat& mask_miss, MetaData& meta, int mode);
  float augmentation_rotate(cv::Mat& img_src, cv::Mat& img_aug, cv::Mat& mask_miss, MetaData& meta, int mode);
  float augmentation_scale(cv::Mat& img, cv::Mat& img_temp, cv::Mat& mask_miss, MetaData& meta, int mode);
  cv::Size augmentation_croppad(cv::Mat& img_temp, cv::Mat& img_aug, cv::Mat& mask_miss, cv::Mat& mask_miss_aug, MetaData& meta, int mode);

  void RotatePoint(cv::Point2f& p, cv::Mat R);
  bool onPlane(cv::Point p, cv::Size img_size);
  void swapLeftRight(Joints& j);

  int np_in_lmdb;
  int np;
  bool is_table_set;
  vector<vector<float> > aug_degs;
  vector<vector<int> > aug_flips;

 protected:
   /**
   * @brief Generates a random integer from Uniform({0, 1, ..., n-1}).
   *
   * @param n
   *    The upperbound (exclusive) value of the random number.
   * @return
   *    A uniformly random integer value from ({0, 1, ..., n-1}).
   */
  virtual int Rand(int n);

  cv::Mat ReadImageStreamToCVMat(vector<unsigned char>& imbuf, const int height, const int width, const bool is_color);
  void ReadMetaData2(MetaData& meta, const std::string &data);
  void TransformMetaJoints(MetaData& meta);
  void TransformJoints(Joints& joints);
  void clahe(cv::Mat& img, int, int);
  void putGaussianMaps(Dtype* entry, cv::Point2f center, int stride, int grid_x, int grid_y, float sigma);
  void putVecMaps(Dtype* entryX, Dtype* entryY, cv::Mat& count, cv::Point2f centerA, cv::Point2f centerB, int stride, int grid_x, int grid_y, float sigma, int thre);

  // Tranformation parameters
  CPMTransformationParameter param_;

  shared_ptr<Caffe::RNG> rng_;
  Phase phase_;
  Blob<Dtype> data_mean_;
  vector<Dtype> mean_values_;
};

}  // namespace caffe

#endif  // CAFFE_CPM_DATA_TRANSFORMER_HPP_
