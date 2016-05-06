#include "stdafx.h"

using namespace std;
using namespace System;
using namespace System::Runtime::InteropServices;
using namespace System::Collections::Generic;
using namespace System::IO;
using namespace System::Drawing;
using namespace System::Drawing::Imaging;

#define TO_NATIVE_STRING(str) msclr::interop::marshal_as<std::string>(str)
#define MARSHAL_ARRAY(n_array, m_array) \
  auto m_array = gcnew array<float>(n_array.Size); \
  pin_ptr<float> pma = &m_array[0]; \
  memcpy(pma, n_array.Data, n_array.Size * sizeof(float));

namespace CaffeLibMC {

    public ref class CaffeModel
    {
    private:
        _CaffeModel *m_net;
        String ^_netFile;

        string ConvertToDatum(Bitmap ^imgData)
        {
            string datum_string;

            int width = m_net->GetInputImageWidth();
            int height = m_net->GetInputImageHeight();

            Drawing::Rectangle rc = Drawing::Rectangle(0, 0, width, height);

            // resize image
            Bitmap ^temp_bmp = gcnew Bitmap((Image ^)imgData, width, height);
            Bitmap ^resizedBmp = temp_bmp->Clone(rc, PixelFormat::Format24bppRgb);
            delete temp_bmp;

            // get image data block
            BitmapData ^bmpData = resizedBmp->LockBits(rc, ImageLockMode::ReadOnly, resizedBmp->PixelFormat);
            pin_ptr<char> bmpBuffer = (char *)bmpData->Scan0.ToPointer();

            // prepare string buffer to call Caffe model
            datum_string.resize(3 * width * height);
            char *buff = &datum_string[0];
            for (int c = 0; c < 3; ++c)
            {
                for (int h = 0; h < height; ++h)
                {
                    int line_offset = h * bmpData->Stride + c;
                    for (int w = 0; w < width; ++w)
                    {
                        *buff++ = bmpBuffer[line_offset + w * 3];
                    }
                }
            }
            resizedBmp->UnlockBits(bmpData);
            delete resizedBmp;

            return datum_string;
        }

    public:
        static int DeviceCount;

        static CaffeModel()
        {
            int count;
            cudaGetDeviceCount(&count);
            DeviceCount = count;
        }

        static void SetDevice(int deviceId)
        {
            _CaffeModel::SetDevice(deviceId);
        }

        CaffeModel(String ^netFile, String ^modelFile)
        {
            _netFile = Path::GetFullPath(netFile);
            m_net = new _CaffeModel(TO_NATIVE_STRING(netFile), TO_NATIVE_STRING(modelFile));
        }

        CaffeModel(CaffeModel ^other)
        {
            String ^netFile = other->_netFile;
            m_net = new _CaffeModel(TO_NATIVE_STRING(netFile), other->m_net);
        }

        // destructor to call finalizer
        ~CaffeModel()
        {
            this->!CaffeModel();
        }

        // finalizer to release unmanaged resource
        !CaffeModel()
        {
            delete m_net;
            m_net = NULL;
        }

        String^ GetNetFileName()
        {
            return _netFile;
        }

        int GetInputImageNum()
        {
            return m_net->GetInputImageNum();
        }

        void SetMeanFile(String ^meanFile)
        {
            m_net->SetMeanFile(TO_NATIVE_STRING(Path::GetFullPath(meanFile)));
        }

        void SetMeanValue(array<float> ^meanValue)
        {
            vector<float> mean_value(meanValue->Length);
            pin_ptr<float> pma = &meanValue[0];
            memcpy(&mean_value[0], pma, meanValue->Length * sizeof(float));
            m_net->SetMeanValue(mean_value);
        }

        array<String^>^ GetLayerNames()
        {
            vector<string> names = m_net->GetLayerNames();
            auto outputs = gcnew array<String^>(names.size());
            for (int i = 0; i < names.size(); i++)
                outputs[i] = gcnew String(names[i].c_str());
            return outputs;
        }

        array<array<float>^>^ GetParams(String^ layerName)
        {
            auto params = m_net->GetParams(TO_NATIVE_STRING(layerName));

            auto outputs = gcnew array<array<float>^>(params.size());
            for (int i = 0; i < params.size(); ++i)
            {
                MARSHAL_ARRAY(params[i], values)
                outputs[i] = values;
            }
            return outputs;

        }

        void SetParams(String^ layerName, array<array<float>^>^ param_blobs)
        {
            vector<FloatArray> native_params(param_blobs->Length);
            // just pin one element to pin the entire object, according to:
            // https://msdn.microsoft.com/en-us/library/18132394.aspx
            pin_ptr<float> pin_params = &param_blobs[0][0];
            for (int i = 0; i < param_blobs->Length; ++i)
            {
                pin_ptr<float> pinned_buffer = &param_blobs[i][0];
                native_params[i].Data = pinned_buffer;
                native_params[i].Size = param_blobs[i]->Length;
            }
            m_net->SetParams(TO_NATIVE_STRING(layerName), native_params);
        }

        void SaveModel(String^ modelFile)
        {
            m_net->SaveModel(TO_NATIVE_STRING(modelFile));
        }

        array<float>^ ExtractOutputs(array<Bitmap^> ^imgData, String^ blobName)
        {
            vector<string> datums(imgData->Length);
            for (int i = 0; i < imgData->Length; i++)
                datums[i] = ConvertToDatum(imgData[i]);

            FloatArray intermediate = m_net->ExtractBitmapOutputs(datums, 0, TO_NATIVE_STRING(blobName));
            MARSHAL_ARRAY(intermediate, outputs)
                return outputs;
        }

        array<array<float>^>^ ExtractOutputs(array<Bitmap^> ^imgData, array<String^>^ blobNames)
        {
            vector<string> datums(imgData->Length);
            for (int i = 0; i < imgData->Length; i++)
                datums[i] = ConvertToDatum(imgData[i]);

            vector<string> names;
            for each(String^ name in blobNames)
                names.push_back(TO_NATIVE_STRING(name));
            vector<FloatArray> intermediates = m_net->ExtractBitmapOutputs(datums, 0, names);
            auto outputs = gcnew array<array<float>^>(names.size());
            for (int i = 0; i < names.size(); ++i)
            {
                auto intermediate = intermediates[i];
                MARSHAL_ARRAY(intermediate, values)
                    outputs[i] = values;
            }
            return outputs;
        }
    };

}
