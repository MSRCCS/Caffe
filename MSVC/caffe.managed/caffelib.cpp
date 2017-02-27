#include "stdafx.h"

using namespace System;
using namespace System::Runtime::InteropServices;
using namespace System::Collections::Generic;
using namespace System::IO;
using namespace System::Drawing;
using namespace System::Drawing::Imaging;

using std::vector;
using std::string;

#define TO_NATIVE_STRING(str) msclr::interop::marshal_as<std::string>(str)
#define MARSHAL_ARRAY(n_array, m_array) \
  auto m_array = gcnew array<float>(n_array.Size); \
  pin_ptr<float> pma = &m_array[0]; \
  memcpy(pma, n_array.Data, n_array.Size * sizeof(float));
#define MARSHAL_VECTOR(vec, m_array) \
  auto m_array = gcnew array<int>(vec.size()); \
  pin_ptr<int> pma = &m_array[0]; \
  memcpy(pma, &vec[0], vec.size() * sizeof(int));

namespace CaffeLibMC {

    public ref class CaffeModel
    {
    private:
        _CaffeModel *m_net;
        String ^_netFile;
        int _resize_target;
        bool _keep_aspect_ratio;

        Bitmap^ ResizeImage(Bitmap ^imgData)
        {
            int width, height;

            if (m_net->HasMeanFile())
            {
                m_net->GetMeanFileResolution(width, height);
            }
            else // check _resize_target
            {
                if (!_keep_aspect_ratio)
                {
                    width = _resize_target;
                    height = _resize_target;
                }
                else
                {
                    int ori_size = Math::Min(imgData->Width, imgData->Height);
                    width = (int)((float)imgData->Width * _resize_target / ori_size + 0.5f);
                    height = (int)((float)imgData->Height * _resize_target / ori_size + 0.5f);
                }
            }

            Drawing::Rectangle rc = Drawing::Rectangle(0, 0, width, height);

            // resize image
            Bitmap ^resizedBmp;
            if (width == imgData->Width && height == imgData->Height)
            {
                resizedBmp = imgData;
            }
            else
            {
                resizedBmp = gcnew Bitmap((Image ^)imgData, width, height);
                resizedBmp = resizedBmp->Clone(rc, PixelFormat::Format24bppRgb);
            }

            return resizedBmp;
        }

        string ConvertToDatum(Bitmap ^imgData, bool resizeImage, int &new_width, int &new_height)
        {
            Bitmap ^img;
            if (resizeImage)
                img = ResizeImage(imgData);
            else
                img = imgData;

            new_width = img->Width;
            new_height = img->Height;

            Drawing::Rectangle rc = Drawing::Rectangle(0, 0, new_width, new_height);

            if (img->PixelFormat != PixelFormat::Format24bppRgb)
            {
                // here img is actually pointing to imgData, and imgData is not of Format24bppRgb.
                img = gcnew Bitmap((Image ^)imgData, img->Width, img->Height);
                img = img->Clone(rc, PixelFormat::Format24bppRgb);
            }

            // get image data block
            BitmapData ^bmpData = img->LockBits(rc, ImageLockMode::ReadOnly, img->PixelFormat);
            pin_ptr<char> bmpBuffer = (char *)bmpData->Scan0.ToPointer();

            // prepare string buffer to call Caffe model
            string datum_string;
            datum_string.resize(3 * new_width * new_height);
            char *buff = &datum_string[0];
            for (int c = 0; c < 3; ++c)
            {
                for (int h = 0; h < new_height; ++h)
                {
                    int line_offset = h * bmpData->Stride + c;
                    for (int w = 0; w < new_width; ++w)
                    {
                        *buff++ = bmpBuffer[line_offset + w * 3];
                    }
                }
            }
            img->UnlockBits(bmpData);

            if (img != imgData)
                delete img;

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
            _resize_target = 0;
            _keep_aspect_ratio = false;
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

        // By default, the mean file specifies the input image size, which should be the same as mean file resolution.
        void SetMeanFile(String ^meanFile)
        {
            m_net->SetMeanFile(TO_NATIVE_STRING(Path::GetFullPath(meanFile)));
        }

        // To use mean value, the input image size and crop type must be specified by calling SetResizeTarget(...).
        void SetMeanValue(array<float> ^meanValue, bool do_crop)
        {
            vector<float> mean_value(meanValue->Length);
            pin_ptr<float> pma = &meanValue[0];
            memcpy(&mean_value[0], pma, meanValue->Length * sizeof(float));
            m_net->SetMeanValue(mean_value, do_crop);
        }

        // if keep_aspect_ratio == true, image shorter size will be resized to resize_target.
        // otherwise, image will be resized to (resize_target, resize_target).
        void SetResizeTarget(int resize_target, bool keep_aspect_ratio)
        {
            _resize_target = resize_target;
            _keep_aspect_ratio = keep_aspect_ratio;
        }

        array<int>^ GetBlobShape(String ^blobName)
        {
            vector<int> shape = m_net->GetBlobShape(TO_NATIVE_STRING(blobName));
            MARSHAL_VECTOR(shape, outputs);
            return outputs;
        }

        void SetInputBatchSize(int batch_size)
        {
            m_net->SetInputBatchSize(batch_size);
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

        array<float>^ ExtractOutputs(array<Bitmap^> ^imgData, String^ blobName, bool resizeImage)
        {
            vector<string> datums(imgData->Length);
            vector<int> imgSize(imgData->Length * 2); // to store (width, height) for each image
            for (int i = 0; i < imgData->Length; i++)
                datums[i] = ConvertToDatum(imgData[i], resizeImage, imgSize[i * 2 + 0], imgSize[i * 2 + 1]);

            if (!resizeImage)
            {
                // Reshape input blob to fit input image
                // and assume all the images have the same size
                m_net->SetInputResolution(imgData[0]->Width, imgData[0]->Height);
            }

            FloatArray intermediate = m_net->ExtractBitmapOutputs(datums, imgSize, TO_NATIVE_STRING(blobName));
            MARSHAL_ARRAY(intermediate, outputs)
                return outputs;
        }

        array<array<float>^>^ ExtractOutputs(array<Bitmap^> ^imgData, array<String^>^ blobNames, bool resizeImage)
        {
            vector<string> datums(imgData->Length);
            vector<int> imgSize(imgData->Length * 2); // to store (width, height) for each image
            for (int i = 0; i < imgData->Length; i++)
                datums[i] = ConvertToDatum(imgData[i], resizeImage, imgSize[i * 2 + 0], imgSize[i * 2 + 1]);

            if (!resizeImage)
            {
                // Reshape input blob to fit input image
                // and assume all the images have the same size
                m_net->SetInputResolution(imgData[0]->Width, imgData[0]->Height);
            }

            vector<string> names;
            for each(String^ name in blobNames)
                names.push_back(TO_NATIVE_STRING(name));
            vector<FloatArray> intermediates = m_net->ExtractBitmapOutputs(datums, imgSize, names);
            auto outputs = gcnew array<array<float>^>(names.size());
            for (int i = 0; i < names.size(); ++i)
            {
                auto intermediate = intermediates[i];
                MARSHAL_ARRAY(intermediate, values)
                    outputs[i] = values;
            }
            return outputs;
        }

        void SetInputs(String^ blobName, array<Bitmap^>^ imgData, bool resizeImage)
        {
            vector<string> datums(imgData->Length);
            vector<int> imgSize(imgData->Length * 2); // to store (width, height) for each image
            for (int i = 0; i < imgData->Length; i++)
                datums[i] = ConvertToDatum(imgData[i], resizeImage, imgSize[i * 2 + 0], imgSize[i * 2 + 1]);

            if (!resizeImage)
            {
                // Reshape input blob to fit input image
                // and assume all the images have the same size
                m_net->SetInputResolution(imgData[0]->Width, imgData[0]->Height);
            }

            m_net->SetInputs(TO_NATIVE_STRING(blobName), datums, imgSize);
        }

        void SetInputs(String^ blobName, array<float>^ data)
        {
            vector<float> vec(data->Length);
            pin_ptr<float> pma = &data[0];
            memcpy(&vec[0], pma, vec.size() * sizeof(float));
            m_net->SetInputs(TO_NATIVE_STRING(blobName), vec);
        }

        array<float>^ Forward(String^ outputBlobName)
        {
            FloatArray intermediate = m_net->Forward(TO_NATIVE_STRING(outputBlobName));
            MARSHAL_ARRAY(intermediate, outputs);
            return outputs;
        }

        array<array<float>^>^ Forward(array<String^>^ outputBlobNames)
        {
            vector<string> names;
            for each(String^ name in outputBlobNames)
                names.push_back(TO_NATIVE_STRING(name));
            vector<FloatArray> intermediates = m_net->Forward(names);
            auto outputs = gcnew array<array<float>^>(names.size());
            for (int i = 0; i < names.size(); ++i)
            {
                auto intermediate = intermediates[i];
                MARSHAL_ARRAY(intermediate, values);
                outputs[i] = values;
            }
            return outputs;
        }
    };
}
