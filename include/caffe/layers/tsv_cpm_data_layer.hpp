#ifndef CAFFE_TSV_CPM_DATA_LAYER_HPP_
#define CAFFE_TSV_CPM_DATA_LAYER_HPP_

#include <vector>
#include <map>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/tsv_data_layer.hpp"
#include "caffe/cpm_data_transformer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides box data to the Net from tsv files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class TsvCPMDataLayer : public TsvDataLayer<Dtype> {
public:
	explicit TsvCPMDataLayer(const LayerParameter& param);
        //: TsvDataLayer<Dtype>(param) {}
	virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "TsvCPMData"; }
	virtual inline int ExactNumBottomBlobs() const { return 0; }
	virtual inline int MinTopBlobs() const { return 1; }
	virtual inline int MaxTopBlobs() const { return 2; }

protected:
    virtual void process_one_image_and_label(const string &input_b64coded_data, const string &input_label_data, const TsvDataParameter &tsv_param, Dtype *output_image_data, Dtype *output_label_data);

private:
	CPMTransformationParameter cpm_transform_param_;
	shared_ptr<CPMDataTransformer<Dtype> > cpm_data_transformer_;
};

}  // namespace caffe

#endif  // CAFFE_TSV_CPM_DATA_LAYER_HPP_
