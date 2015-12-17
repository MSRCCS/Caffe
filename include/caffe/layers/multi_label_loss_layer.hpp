#ifndef CAFFE_MULTI_LABEL_LOSS_LAYER_HPP_
#define CAFFE_MULTI_LABEL_LOSS_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class MultiLabelLossLayer : public LossLayer<Dtype> {
public:
	explicit MultiLabelLossLayer(const LayerParameter& param)
		: LossLayer<Dtype>(param) {}
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "MultiLabelLoss"; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	/// prob stores the output probability predictions for each class
	Blob<Dtype> prob_pos_;
	Blob<Dtype> prob_neg_;
	/// Whether to normalize the loss by the total number of values present
	/// (otherwise just by the batch size).
	bool normalize_;
};

}  // namespace caffe

#endif  // CAFFE_MULTI_LABEL_LOSS_LAYER_HPP_
