// Due to a bug caused by C++/CLI and boost (used indirectly via caffe headers, not this one), 
// we have to seperate code related to boost from CLI compiling environment.
// This wrapper class serves for this purpose.
// See: http://stackoverflow.com/questions/8144630/mixed-mode-c-cli-dll-throws-exception-on-exit
//	and http://article.gmane.org/gmane.comp.lib.boost.user/44515/match=string+binding+invalid+mixed

#pragma once

#include <string>
#include <vector>

//Declare an abstract Net class instead of including caffe headers, which include boost headers.
//The definition of Net is defined in cpp code, which does include caffe header files.
namespace caffe
{
    template <class DType>
    class Net;
    template <class DType>
    class DataTransformer;
}

struct FloatArray
{
    const float* Data;
    int Size;
    FloatArray();
    FloatArray(const float* data, int size);
};

typedef std::vector<float> FloatVec;

class _CaffeModel
{
    caffe::Net<float>* _net;

    caffe::DataTransformer<float>* _data_transformer;
    std::vector<float> _mean_value;
    bool _do_crop;          // _do_crop is only valid when _mean_value is set
    std::string _mean_file;
    int _data_mean_width;   // _data_mean_width and _data_mean_height are only valid when _mean_file is set
    int _data_mean_height;

    void EvaluateBitmap(const std::vector<std::string> &imageData, const std::vector<int> &imgSize);

public:
    static void SetDevice(int device_id); //Use a negative number for CPU only

    _CaffeModel(const std::string &netFile, const std::string &modelFile);
    _CaffeModel(const std::string &netFile, _CaffeModel *other);
    ~_CaffeModel();

    // only set mean file or mean value, but not both
    void SetMeanFile(const std::string &meanFile);
    void SetMeanValue(const std::vector<float> &meanValue, bool do_crop);
    bool HasMeanFile();
    bool HasMeanValue();

    void GetMeanFileResolution(int &width, int &height);

    int GetInputImageNum();
    std::vector<int> GetBlobShape(const std::string blobName);
    void _CaffeModel::ReshapeBlob(const std::string blobName, const std::vector<int> shape);
    void SetInputResolution(int width, int height);
    void SetInputBatchSize(int batch_size);

    std::vector<std::string> GetLayerNames();
    std::vector<FloatArray> GetParams(const std::string layerName);
    void SetParams(const std::string layerName, std::vector<FloatArray>& params);
    void SaveModel(const std::string modelFile);

    // This function assumes the images are of the same size for both width and height.
    // Typical calling cases:
    //   For object detection: passing a single image with the specified width and height;
    //   For image classification: passing multiple images with the same width and height;
    void SetInputs(const std::string &blobName, const std::vector<std::string> &imageData, const std::vector<int> &imgSize);
    void SetInputs(const std::string &blobName, const std::vector<float> &data);

    FloatArray Forward(const std::string &outputBlobName);
    std::vector<FloatArray> Forward(const std::vector<std::string> &outputBlobNames);

    // This function assumes the images are of the same size for both width and height.
    FloatArray ExtractBitmapOutputs(const std::vector<std::string> &imageData, const std::vector<int> &imgSize, const std::string &layerName);
    // This function assumes the images are of the same size for both width and height.
    std::vector<FloatArray> ExtractBitmapOutputs(const std::vector<std::string> &imageData, const std::vector<int> &imgSize, const std::vector<std::string> &layerNames);
};
