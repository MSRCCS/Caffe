#ifndef CAFFE_DATA_LAYER_HPP_
#define CAFFE_DATA_LAYER_HPP_

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
class thread_closure;
template <typename Dtype>
class TsvDataLayer : public BasePrefetchingDataLayer<Dtype> {
public:
	explicit TsvDataLayer(const LayerParameter& param)
		: BasePrefetchingDataLayer<Dtype>(param) {}
	virtual ~TsvDataLayer();
	virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "TsvData"; }
	virtual inline int ExactNumBottomBlobs() const { return 0; }
	virtual inline int MinTopBlobs() const { return 1; }
	virtual inline int MaxTopBlobs() const { return 2; }

protected:
	void transform_datum(thread_closure<Dtype> &c, size_t dst_index);
	virtual void load_batch(Batch<Dtype>* batch);
	TsvRawDataFile tsv_;
	TsvRawDataFile tsv_label_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYER_HPP_
