#include <opencv2/core/core.hpp>

#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layers/tsv_cpm_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/random_helper.h"

#define BOOST_SPIRIT_THREADSAFE
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>

using boost::property_tree::ptree;
using boost::property_tree::read_json;

namespace caffe {

	template <typename Dtype>
	TsvCPMDataLayer<Dtype>::TsvCPMDataLayer(const LayerParameter& param)
		: TsvDataLayer<Dtype>(param),
		cpm_transform_param_(param.cpm_transform_param()) {
	}

	template <typename Dtype>
	void TsvCPMDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		
		cpm_data_transformer_.reset(
			new CPMDataTransformer<Dtype>(cpm_transform_param_, this->phase_));
		cpm_data_transformer_->InitRand();
		
		// specify the shape of top[0] & top[1]
		TsvDataLayer<Dtype>::DataLayerSetUp(bottom, top);
		
		// reshape label
		vector<int> shape = top[1]->shape();
		const int stride = this->layer_param_.cpm_transform_param().stride();
		const int height = this->layer_param_.cpm_transform_param().crop_size_y();
		const int width = this->layer_param_.cpm_transform_param().crop_size_x();

		const int num_parts = this->layer_param_.cpm_transform_param().num_parts();
		top[1]->Reshape(shape[0], 2 * (num_parts + 1), height / stride, width / stride);
		for (int i = 0; i < this->prefetch_.size(); ++i) {
			this->prefetch_[i]->label_.Reshape(shape[0], 2 * (num_parts + 1), height / stride, width / stride);
		}
	}

	template <typename Dtype>
	void TsvCPMDataLayer<Dtype>::process_one_image_and_label(const string &input_b64coded_data, const string &input_label_data, const TsvDataParameter &tsv_param, Dtype *output_image_data, Dtype *output_label_data)
	{
		this->cpm_data_transformer_->Transform_nv2(input_b64coded_data, input_label_data,
			output_image_data, output_label_data, tsv_param.data_format() == TsvDataParameter_DataFormat_ImagePath);
	}

	INSTANTIATE_CLASS(TsvCPMDataLayer);
	REGISTER_LAYER_CLASS(TsvCPMData);

}  // namespace caffe

