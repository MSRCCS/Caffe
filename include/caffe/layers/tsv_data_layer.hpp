#ifndef CAFFE_TSV_DATA_LAYER_HPP_
#define CAFFE_TSV_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/tsv_data_io.hpp"

namespace caffe {

cv::Mat ReadImageStreamToCVMat(vector<unsigned char>& imbuf, const int height, const int width, const bool is_color);

/**
 * @brief Provides data to the Net from tsv files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class TsvDataLayer : public BasePrefetchingDataLayer<Dtype> {
public:
	explicit TsvDataLayer(const LayerParameter& param)
		: BasePrefetchingDataLayer<Dtype>(param), offset_() {}
	virtual ~TsvDataLayer();
	virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "TsvData"; }
	virtual inline int ExactNumBottomBlobs() const { return 0; }
	virtual inline int MinTopBlobs() const { return 1; }
	virtual inline int MaxTopBlobs() const { return 2; }

protected:
    virtual void process_one_image_and_label(const string &input_b64coded_data, const string &input_label_data, const TsvDataParameter &tsv_param, Dtype *output_image_data, Dtype *output_label_data);
    virtual void Next();
    virtual bool Skip();
    virtual void load_batch(Batch<Dtype>* batch);
    virtual void on_load_batch(Batch<Dtype>* batch);

    TsvRawDataFile tsv_;
	TsvRawDataFile tsv_label_;
    uint64_t offset_;

    // mean values for pixel value subtraction
    std::vector<Dtype> mean_values_;

private:
    void load_kl(const string &kl_filename);
    void get_random_kl_shift(std::vector<float> &shift, float kl_coef);
    cv::Rect get_crop_rect(const cv::Mat &img, const TsvDataParameter &tsv_param);
    void CVMatToBlobBuffer(const cv::Mat &cv_img_float, Dtype *buffer);

    void process_one_image(const string &input_b64coded_data, const TsvDataParameter &tsv_param, Dtype *output_image_data);
    void process_one_label(const string &input_label_data, const TsvDataParameter &tsv_param, Dtype *output_label_data);

    // kl eigen values and vectors for color jittering
    std::vector<Dtype> eig_val_;    // vector of the 3 eigen values
    std::vector<Dtype> eig_vec_;    // vector of 9 values for a 3x3 matrix, the first 3 values are for the first eigen vector, and so on.
};

}  // namespace caffe

#endif  // CAFFE_TSV_DATA_LAYER_HPP_
