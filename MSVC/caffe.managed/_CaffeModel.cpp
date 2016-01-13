#include "_CaffeModel.h"

#include <cuda_runtime.h>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <stdio.h>
#include "caffe/caffe.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "boost/smart_ptr/shared_ptr.hpp"

using namespace boost;
using namespace caffe;

FloatArray::FloatArray(const float* data, int size) : Data(data), Size(size) {}

_CaffeModel::_CaffeModel()
{
	_net = NULL;
}


_CaffeModel::~_CaffeModel()
{
	if (_net)
	{
		delete _net;
		_net = NULL;
	}
}

void _CaffeModel::SetDevice(int deviceId)
{
    // Set GPU
    if (deviceId >= 0)
    {
        Caffe::set_mode(Caffe::GPU);
        Caffe::SetDevice(deviceId);
    }
    else
        Caffe::set_mode(Caffe::CPU);
}

int _CaffeModel::GetInputImageWidth()
{
	MemoryDataLayer<float> * layer = (MemoryDataLayer<float>*)_net->layer_by_name("data").get();
	return layer->width();
}

int _CaffeModel::GetInputImageHeight()
{
	MemoryDataLayer<float> * layer = (MemoryDataLayer<float>*)_net->layer_by_name("data").get();
	return layer->height();
}

int _CaffeModel::GetInputImageChannels()
{
	MemoryDataLayer<float> * layer = (MemoryDataLayer<float>*)_net->layer_by_name("data").get();
	return layer->channels();
}

void _CaffeModel::Init(const std::string &netFile, const std::string &modelFile)
{
	// Set to TEST Phase
	_net = new Net<float>(netFile, caffe::TEST);
	_net->CopyTrainedLayersFrom(modelFile);
}

void _CaffeModel::ExtractFeatureFromFile(const std::string &imageFile, const std::string &blobName, std::vector<float> &feature)
{
	Datum datum;
	ReadImageToDatum(imageFile, 1, GetInputImageWidth(), GetInputImageHeight(), &datum);

	std::vector<Datum> datums;
	for (int i = 0; i < 1; i++)
		datums.push_back(datum);

	// Net initialization
	float loss = 0.0;
	shared_ptr<MemoryDataLayer<float> > memory_data_layer;
	memory_data_layer = static_pointer_cast<MemoryDataLayer<float>>(_net->layer_by_name("data"));
	memory_data_layer->AddDatumVector(datums);

	const std::vector<Blob<float>*>& results = _net->ForwardPrefilled(&loss);

	const shared_ptr<Blob<float>> featureBlob = _net->blob_by_name(blobName);
	feature.resize(featureBlob->count());
	memcpy(&feature[0], featureBlob->cpu_data(), featureBlob->count() * sizeof(float));
}

void Evaluate(caffe::Net<float>* net, const string &imageData, int interpolation)
{
	// Net initialization
	float loss = 0.0;
	shared_ptr<MemoryDataLayer<float> > memory_data_layer;
	memory_data_layer = static_pointer_cast<MemoryDataLayer<float>>(net->layer_by_name("data"));

	Datum datum;
	datum.set_channels(3);
	datum.set_height(memory_data_layer->height());
	datum.set_width(memory_data_layer->width());
	datum.set_label(0);
	datum.clear_data();
	datum.clear_float_data();
	datum.set_data(imageData);

	std::vector<Datum> datums;
	for (int i = 0; i < 1; i++)
		datums.push_back(datum);

	memory_data_layer->AddDatumVector(datums);
	const std::vector<Blob<float>*>& results = net->ForwardPrefilled(&loss);

}

FloatArray _CaffeModel::ExtractOutputs(const std::string &imageData, int interpolation, const string &blobName)
{
	Evaluate(_net, imageData, interpolation);
	auto blob = _net->blob_by_name(blobName);
	return FloatArray(blob->cpu_data(), blob->count());
}

vector<FloatArray> _CaffeModel::ExtractOutputs(const std::string &imageData, int interpolation, const vector<string> &layerNames)
{
	Evaluate(_net, imageData, interpolation);
	vector<FloatArray> results;
	for (auto& name : layerNames)
	{
		auto blob = _net->blob_by_name(name);
		results.push_back(FloatArray(blob->cpu_data(), blob->count()));
	}
	return results;
}