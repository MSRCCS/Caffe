#pragma once

// Due to a bug caused by C++/CLI and boost, we have to seperate code related to boost from CLI compiling environment.
// This wrapper class serves for this purpose.
// See: http://stackoverflow.com/questions/8144630/mixed-mode-c-cli-dll-throws-exception-on-exit
//	and http://article.gmane.org/gmane.comp.lib.boost.user/44515/match=string+binding+invalid+mixed

#include <string>
#include <vector>

namespace caffe
{
	template <class DType>
	class Net;
}

struct FloatArray
{
	const float* Data;
	int Size;
	FloatArray(const float* data, int size);
};

typedef std::vector<float> FloatVec;

class _CaffeModel
{
	caffe::Net<float> *_net;
public:
	_CaffeModel();
	~_CaffeModel();

	int GetInputImageWidth();
	int GetInputImageHeight();
	int GetInputImageChannels();

	void Init(const std::string &netFile, const std::string &modelFile, bool useGpu);

	// legacy api, will be suppressed.
	void ExtractFeatureFromFile(const std::string &imageFile, const std::string &blobName, std::vector<float> &feature);

	// imageData needs to be of size channel*height*width as required by the "data" blob. 
	// The C++/CLI caller can use GetInputImageWidth()/Height/Channels to get the desired dimension.
	FloatArray ExtractOutputs(const std::string &imageData, int interpolation, const std::string &layerName);
	std::vector<FloatArray> ExtractOutputs(const std::string &imageData, int interpolation, const std::vector<std::string> &layerNames);
};

