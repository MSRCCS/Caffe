#pragma once

#ifdef CAFFENATIVE_EXPORTS
#define DLLEXPORT_API __declspec(dllexport)
#else
#define DLLEXPORT_API __declspec(dllimport)
#endif

#include <string>
#include <vector>

namespace caffe
{
template <class DType>
class Net;

struct FloatArray
{
    const float* Data;
    int Size;
    FloatArray(const float* data, int size) : Data(data), Size(size) {};
};

class DLLEXPORT_API CaffeModel
{
    caffe::Net<float> *_net;
public:
    void SetDevice(int device_id); //Use a negative number for CPU only

    CaffeModel();
    ~CaffeModel();

    int GetInputImageWidth();
    int GetInputImageHeight();
    int GetInputImageChannels();

    void Init(const std::string &netFile, const std::string &modelFile);

    // imageData needs to be of size channel*height*width as required by the "data" blob. 
    // The C++ caller can use GetInputImageWidth()/Height/Channels to get the desired dimension.
    FloatArray ExtractOutputs(const std::string &imageData, const std::string &layerName);
    std::vector<FloatArray> ExtractOutputs(const std::string &imageData, const std::vector<std::string> &layerNames);
};

// crop image by rotating the image at the specified center (x,y) and crop the region at 
// (x1, y1, x2, y2), which is (centerX-cropLeft*baseWidth, centerY-cropUp*baseWidth, 
//  centerX+cropRight*baseWidth, centerY+cropDown*baseWidth) 
// The separate of baseWidth is for convenient so that the crop params can be relatively fixed for the same scenario.
void CropImageAndResize(const unsigned char* image, int width, int height, int channels, int stride,
    std::string &dst, int dstWidth, int dstHeight,
    float centerX, float centerY, float ratationRadius,
    float baseWidth, float cropUp, float cropDown, float cropLeft, float cropRight);
}