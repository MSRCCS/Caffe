// caffeLibMC.h

#pragma once

#include "_CaffeModel.h"
#include <cuda_runtime.h> 
#include <msclr\marshal_cppstd.h>

using namespace std;
using namespace System;
using namespace System::Drawing;
using namespace System::Drawing::Imaging;
using namespace System::Runtime::InteropServices;
using namespace System::Collections::Generic;
using namespace System::IO;

#define TO_NATIVE_STRING(str) msclr::interop::marshal_as<string>(str)
#define MARSHAL_ARRAY(n_array, m_array) \
  auto m_array = gcnew array<float>(n_array.Size); \
  pin_ptr<float> pma = &m_array[0]; \
  memcpy(pma, n_array.Data, n_array.Size * sizeof(float));

namespace CaffeLibMC {

	public ref class CaffeModel
	{
	private:
		_CaffeModel *m_net;
		List<String^>^ m_listLabelMap;
		bool m_isInitialized;


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

        CaffeModel()
		{
			m_net = new _CaffeModel();
			m_listLabelMap = gcnew List<String^>();
			m_isInitialized = false;
		}

		// destructor to call finalizer
		~CaffeModel()
		{
			this->!CaffeModel();
		}

		// finalizer to release unmanaged resource
		!CaffeModel()
		{
			if (m_net)
			{
				delete m_net;
				m_net = NULL;
			}
		}

		void Init(String ^netFile, String ^modelFile, String^ label_map)
		{
			if (m_isInitialized)
				return;

			string strNetFile = msclr::interop::marshal_as<string>(netFile);
			string strModelFile = msclr::interop::marshal_as<string>(modelFile);

			m_net->Init(strNetFile, strModelFile);

			StreamReader^ din = File::OpenText(label_map);
			String^ str;
			m_listLabelMap->Clear();
			while ((str = din->ReadLine()) != nullptr)
			{
				m_listLabelMap->Add(str->Split('\t')[0]);
			}

			m_isInitialized = true;
		}

		void ExtractFeatureFromFile(String ^imageFile, String ^blobName, [Runtime::InteropServices::Out] array<float>^ %feature)
		{
			string strImageFile = msclr::interop::marshal_as<string>(imageFile);
			string strBlobName = msclr::interop::marshal_as<string>(blobName);
			vector<float> vecFeature;
			m_net->ExtractFeatureFromFile(strImageFile, strBlobName, vecFeature);
			feature = gcnew array<float>(vecFeature.size());
			pin_ptr<float> pResult = &feature[0];
			memcpy(pResult, &vecFeature[0], vecFeature.size() * sizeof(float));
		}

		string ConvertToDatum(Bitmap ^imgData)
		{
			string datum_string;

			int width = m_net->GetInputImageWidth();
			int height = m_net->GetInputImageHeight();

			Drawing::Rectangle rc = Drawing::Rectangle(0, 0, width, height);

			// resize image
			Bitmap ^resizedBmp;
			if (width == imgData->Width && height == imgData->Height)
			{
				resizedBmp = imgData->Clone(rc, PixelFormat::Format24bppRgb);
			}
			else
			{
				resizedBmp = gcnew Bitmap((Image ^)imgData, width, height);
				resizedBmp = resizedBmp->Clone(rc, PixelFormat::Format24bppRgb);
			}
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

			return datum_string;
		}

		array<float>^ ExtractOutputs(Bitmap^ imgData, String^ blobName)
		{
			string datum_string = ConvertToDatum(imgData);

			FloatArray intermediate = m_net->ExtractOutputs(datum_string, 0, TO_NATIVE_STRING(blobName));
			MARSHAL_ARRAY(intermediate, outputs)
			return outputs;
		}

		array<array<float>^>^ ExtractOutputs(Bitmap^ imgData, array<String^>^ blobNames)
		{
			string datum_string = ConvertToDatum(imgData);

			vector<string> names;
			for each(String^ name in blobNames)
				names.push_back(TO_NATIVE_STRING(name));
			vector<FloatArray> intermediates = m_net->ExtractOutputs(datum_string, 0, names);
			auto outputs = gcnew array<array<float>^>(names.size());
			for (int i = 0; i < names.size(); ++i)
			{
				auto intermediate = intermediates[i];
				MARSHAL_ARRAY(intermediate, values)
					outputs[i] = values;
			}
			return outputs;
		}

		String^ PredictFromFile(String ^imageFile, int topK)
		{
			array<float>^ scores = gcnew array<float>(m_listLabelMap->Count);
			ExtractFeatureFromFile(imageFile, "prob", scores);

			int* topKIndex = new int[topK];
			float* topKScores = new float[topK];

			FindTopK(topKIndex, topKScores, topK, scores);
			delete scores;

			String^ result = "";
			for (int i = 0; i < topK; i++)
			{
				result = result + m_listLabelMap[topKIndex[i]] + ":" + topKScores[i].ToString() + ";";
			}

			return result->TrimEnd(';');
		}

		String^ PredictFromBuffer(Bitmap ^imgData, int topK)
		{
			array<float>^ scores = ExtractOutputs(imgData, "prob");

			int* topKIndex = new int[topK];
			float* topKScores = new float[topK];

			FindTopK(topKIndex, topKScores, topK, scores);
			delete scores;

			String^ result = "";
			for (int i = 0; i < topK; i++)
			{
				result = result + m_listLabelMap[topKIndex[i]] + ":" + topKScores[i].ToString() + ";";
			}

			return result->TrimEnd(';');
		}


		void FindTopK(int* topKIndex, float* topKScores, int K, array<float>^ scores)
		{
			List<Tuple<int, float>^>^ myList = gcnew List<Tuple<int, float>^>();

			int CateNum = m_listLabelMap->Count;

			for (int i = 0; i < CateNum; i++)
			{
				myList->Add(gcnew Tuple<int, float>(i, scores[i]));
			}

			myList->Sort(gcnew Comparison<Tuple<int, float>^>(CompareScoreTuple));

			for (int i = 0; i < min(K, CateNum); i++)
			{
				topKIndex[i] = myList[i]->Item1;
				topKScores[i] = myList[i]->Item2;
			}

			if (K > CateNum)
			{
				for (int i = CateNum; i < K; i++)
				{
					topKIndex[i] = CateNum - 1; // no label (topK is bigger than number of categories)
					topKScores[i] = 0;
				}
			}
		}

		static int CompareScoreTuple(Tuple<int, float>^ x, Tuple<int, float>^ y)
		{
			if (x->Item2 == y->Item2) return 0;
			else if (x->Item2 > y->Item2) return -1;
			else return 1;
		}
	};
}
