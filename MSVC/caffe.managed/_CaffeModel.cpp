#include "_CaffeModel.h"

#pragma warning(push, 0) // disable warnings from the following external headers
#include <vector>
#include <string>
#include <stdio.h>
#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#pragma warning(push, 0) 

using namespace boost;
using namespace caffe;

FloatArray::FloatArray(const float* data, int size) : Data(data), Size(size) {}

_CaffeModel::_CaffeModel(const string &netFile, const string &modelFile)
{
    _net = new Net<float>(netFile, Phase::TEST);
    _net->CopyTrainedLayersFrom(modelFile);

    _mean_file.clear();
    _mean_value.clear();
    _data_transformer = NULL;
    _data_mean_width = 0;
    _data_mean_height = 0;
}

_CaffeModel::_CaffeModel(const std::string &netFile, _CaffeModel *other)
{
    _net = new Net<float>(netFile, Phase::TEST);
    _net->ShareTrainedLayersWith(other->_net);

    _mean_file.clear();
    _mean_value.clear();
    _data_transformer = NULL;
    _data_mean_width = 0;
    _data_mean_height = 0;

    if (other->_mean_file.size() > 0)
        this->SetMeanFile(other->_mean_file);
    if (other->_mean_value.size() > 0)
        this->SetMeanValue(other->_mean_value);
}

_CaffeModel::~_CaffeModel()
{
    if (_net)
    {
        delete _net;
        _net = nullptr;
    }
    if (_data_transformer)
    {
        delete _data_transformer;
        _data_transformer = nullptr;
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

void _CaffeModel::SetMeanFile(const std::string &meanFile)
{
    if (_data_transformer)
        delete _data_transformer;

    Blob<float>* input_blob = _net->input_blobs()[0];
    CHECK_EQ(input_blob->width(), input_blob->height()) << "Input blob width (" << input_blob->width()
        << ") and height (" << input_blob->height() << ") should be the same.";

    TransformationParameter transform_param;
    transform_param.set_crop_size(input_blob->width());

    _mean_file = meanFile;

    transform_param.set_mean_file(meanFile.c_str());
    // as data_mean_ in DataTransformer is protected, we load the mean file again to get width and height
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(meanFile.c_str(), &blob_proto);
    Blob<float> data_mean;
    data_mean.FromProto(blob_proto);
    _data_mean_width = data_mean.width();
    _data_mean_height = data_mean.height();

    _data_transformer = new DataTransformer<float>(transform_param, TEST);
    _data_transformer->InitRand();
}

void _CaffeModel::SetMeanValue(const vector<float> &meanValue)
{
    if (_data_transformer)
        delete _data_transformer;

    TransformationParameter transform_param;

    _mean_value = meanValue;
    for (int i = 0; i < meanValue.size(); i++)
        transform_param.add_mean_value(meanValue[i]);

    _data_transformer = new DataTransformer<float>(transform_param, TEST);
    _data_transformer->InitRand();
}

int _CaffeModel::GetInputImageWidth()
{
    if (_mean_file.size() > 0)
        return _data_mean_width;
    Blob<float>* input_blob = _net->input_blobs()[0];
    return input_blob->width();
}

int _CaffeModel::GetInputImageHeight()
{
    if (_mean_file.size() > 0)
        return _data_mean_height;
    Blob<float>* input_blob = _net->input_blobs()[0];
    return input_blob->height();
}

int _CaffeModel::GetInputImageChannels()
{
    Blob<float>* input_blob = _net->input_blobs()[0];
    return input_blob->channels();
}

void _CaffeModel::EvaluateBitmap(const string &imageData, int interpolation)
{
    Blob<float>* input_blob = _net->input_blobs()[0];
    float* input_data = input_blob->mutable_cpu_data();
    if (_data_transformer)
    {
        Datum datum;
        datum.set_channels(3);
        datum.set_height(this->GetInputImageHeight());
        datum.set_width(this->GetInputImageWidth());
        datum.set_label(0);
        datum.clear_data();
        datum.clear_float_data();
        datum.set_data(imageData);

        _data_transformer->Transform(datum, input_blob);
    }
    else
    {
        Blob<float>* input_blob = _net->input_blobs()[0];
        int height = input_blob->height();
        int width = input_blob->width();

        // imageData is already in the format of c*h*w
        BYTE * src_data = (BYTE *)&imageData[0];
        for (int i = 0; i < 3; ++i)
        {
            *input_data = (float)src_data[i];
        }

    }

    float loss = 0.0;
    _net->ForwardPrefilled(&loss);

}

FloatArray _CaffeModel::ExtractBitmapOutputs(const std::string &imageData, int interpolation, const string &blobName)
{
    EvaluateBitmap(imageData, interpolation);
    auto blob = _net->blob_by_name(blobName);
    return FloatArray(blob->cpu_data(), blob->count());
}

vector<FloatArray> _CaffeModel::ExtractBitmapOutputs(const std::string &imageData, int interpolation, const vector<string> &layerNames)
{
    EvaluateBitmap(imageData, interpolation);
    vector<FloatArray> results;
    for (auto& name : layerNames)
    {
        auto blob = _net->blob_by_name(name);
        results.push_back(FloatArray(blob->cpu_data(), blob->count()));
    }
    return results;
}