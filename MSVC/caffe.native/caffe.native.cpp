// caffe.native.dll.cpp : Defines the exported functions for the DLL application.
//
#include "caffe.native.h"
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
#include <opencv2/imgproc/imgproc.hpp>
#include "boost/smart_ptr/shared_ptr.hpp"
#include "caffe/util/io.hpp"
#include <direct.h>

using namespace boost;
namespace caffe
{

CaffeModel::CaffeModel()
{
    _net = NULL;
}

CaffeModel::~CaffeModel()
{
    if (_net)
    {
        delete _net;
        _net = NULL;
    }
}

void CaffeModel::SetDevice(int deviceId)
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

int CaffeModel::GetInputImageWidth()
{
    MemoryDataLayer<float> * layer = (MemoryDataLayer<float>*)_net->layer_by_name("data").get();
    return layer->width();
}

int CaffeModel::GetInputImageHeight()
{
    MemoryDataLayer<float> * layer = (MemoryDataLayer<float>*)_net->layer_by_name("data").get();
    return layer->height();
}

int CaffeModel::GetInputImageChannels()
{
    MemoryDataLayer<float> * layer = (MemoryDataLayer<float>*)_net->layer_by_name("data").get();
    return layer->channels();
}

void CaffeModel::Init(const std::string &netFile, const std::string &modelFile)
{
    char curdir[FILENAME_MAX];
    getcwd(curdir, FILENAME_MAX);
    size_t pos = netFile.find_last_of('\\');
    if (pos != string::npos)
    {
        string netdir = netFile.substr(0, pos);
        chdir(netdir.c_str());
    }
    // Set to TEST Phase
    _net = new Net<float>(netFile, caffe::TEST);
    _net->CopyTrainedLayersFrom(modelFile);
    if (pos != string::npos)
        chdir(curdir);
}

void Evaluate(caffe::Net<float>* net, const string &imageData)
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

FloatArray CaffeModel::ExtractOutputs(const std::string &imageData, const string &blobName)
{
    Evaluate(_net, imageData);
    auto blob = _net->blob_by_name(blobName);
    return FloatArray(blob->cpu_data(), blob->count());
}

vector<FloatArray> CaffeModel::ExtractOutputs(const std::string &imageData, const vector<string> &layerNames)
{
    Evaluate(_net, imageData);
    vector<FloatArray> results;
    for (auto& name : layerNames)
    {
        auto blob = _net->blob_by_name(name);
        results.push_back(FloatArray(blob->cpu_data(), blob->count()));
    }
    return results;
}

}