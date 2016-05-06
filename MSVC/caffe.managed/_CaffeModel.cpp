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

FloatArray::FloatArray() : Data(NULL), Size(0) {}
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

int _CaffeModel::GetInputImageNum()
{
    Blob<float>* input_blob = _net->input_blobs()[0];
    return input_blob->num();
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

std::vector<string> _CaffeModel::GetLayerNames()
{
    return _net->layer_names();
}

std::vector<FloatArray> _CaffeModel::GetParams(const std::string layerName)
{
    Layer<float>* layer = _net->layer_by_name(layerName).get();
    CHECK(layer != NULL) << "Cannot find layer " << layerName;

    vector<FloatArray> results;
    for (auto& blob : layer->blobs())
    {
        results.push_back(FloatArray(blob.get()->cpu_data(), blob.get()->count()));
    }
    return results;
}

void _CaffeModel::SetParams(const std::string layerName, std::vector<FloatArray>& params)
{
    Layer<float>* layer = _net->layer_by_name(layerName).get();
    CHECK(layer != NULL) << "Cannot find layer " << layerName;

    CHECK_EQ(layer->blobs().size(), params.size()) << "Param blob numbers mismatch. Expected: " 
        << layer->blobs().size() << ", Actual: " << params.size();

    for (int i = 0; i < params.size(); i++)
    {
        FloatArray src = params[i];
        Blob<float>* blob = layer->blobs()[i].get();
        CHECK_EQ(src.Size, blob->count()) << "Blob size mismatch. Expected: "
            << blob->count() << ", Actual: " << src.Size;

        caffe_copy<float>(src.Size, src.Data, blob->mutable_cpu_data());
    }
}

void _CaffeModel::SaveModel(const std::string modelFile)
{
    NetParameter net_param;
    _net->ToProto(&net_param, false);
    WriteProtoToBinaryFile(net_param, modelFile);
}

void _CaffeModel::EvaluateBitmap(const std::vector<std::string> &imageData, int interpolation)
{
    Blob<float>* input_blob = _net->input_blobs()[0];

    CHECK_LE(imageData.size(), input_blob->num()) << "Input images (" << imageData.size() 
            << ") should be no more than batch size (" << input_blob->num() << ")";

    if (_data_transformer)
    {
        vector<Datum> datum_vector;
        Datum datum;
        datum.set_channels(3);
        datum.set_height(this->GetInputImageHeight());
        datum.set_width(this->GetInputImageWidth());
        datum.clear_data();
        datum.clear_float_data();

        for (int n = 0; n < imageData.size(); n++)
        {
            datum.set_data(imageData[n]);
            datum_vector.push_back(datum);
        }
        _data_transformer->Transform(datum_vector, input_blob);
    }
    else
    {
        int height = input_blob->height();
        int width = input_blob->width();

        float* input_data = input_blob->mutable_cpu_data();
        float *input_data_ptr = input_data;
        // imageData is already in the format of c*h*w
        for (int n = 0; n < imageData.size(); n++)
        {
            const string &img_data = imageData[n];
            BYTE * src_data = (BYTE *)&img_data[0];
            for (int i = 0; i < input_blob->count(); ++i)
                *input_data_ptr++ = (float)src_data[i];
        }
    }

    float loss = 0.0;
    _net->ForwardPrefilled(&loss);

}

FloatArray _CaffeModel::ExtractBitmapOutputs(const std::vector<std::string> &imageData, int interpolation, const string &blobName)
{
    EvaluateBitmap(imageData, interpolation);
    auto blob = _net->blob_by_name(blobName);
    return FloatArray(blob->cpu_data(), blob->count());
}

vector<FloatArray> _CaffeModel::ExtractBitmapOutputs(const std::vector<std::string> &imageData, int interpolation, const vector<string> &layerNames)
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