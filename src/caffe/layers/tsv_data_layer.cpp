#include <opencv2/core/core.hpp>

#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layers/tsv_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#include <boost/thread.hpp>

//#include <ppl.h>
//using namespace concurrency;

namespace caffe {

cv::Mat ReadImageStreamToCVMat(vector<unsigned char>& imbuf, const int height, const int width, const bool is_color) 
{
	cv::Mat cv_img;
	int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
		CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat cv_img_origin = cv::imdecode(cv::Mat(imbuf), cv_read_flag);
	if (!cv_img_origin.data) {
		LOG(ERROR) << "Could not open memeory image";
		return cv_img_origin;
	}
	if (height > 0 && width > 0) {
		cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
	}
	else {
		cv_img = cv_img_origin;
	}
	return cv_img;
}

cv::Mat CreateCVMat(const unsigned char* image, int width, int height, int channels, int stride)
{
    CHECK(image != NULL && width > 0 && height > 0 && (channels == 1 || channels == 3 || channels == 4)
        && stride >= width * channels) << "Invalid args";

    cv::Mat cv_img;
    if (channels == 1) // 8 bit gray scale palette (special case)
    {
        cv_img = cv::Mat(height, width, CV_8UC1);

        unsigned char* u8_Src = (unsigned char*)image;
        unsigned char* u8_Dst = cv_img.data;

        for (int R = 0; R < height; R++)
        {
            memcpy(u8_Dst, u8_Src, width);
            u8_Src += stride;
            u8_Dst += cv_img.step;
        }
    }
    else // 24 Bit / 32 Bit
    {
        int s32_CvType = channels == 3 ? CV_8UC3 : CV_8UC4;
        // Create a Mat pointing to external memory
        cv::Mat mat(height, width, s32_CvType, (void *)image, stride);

        // Create a Mat with own memory
        mat.copyTo(cv_img);
    }

    return cv_img;
}

#define PI 3.1415926

void CropImageAndResize(const cv::Mat &cv_img,
    std::string &dst, int dstWidth, int dstHeight,
    float centerX, float centerY, float rotationRadius,
    float baseWidth,
    float cropUp, float cropDown, float cropLeft, float cropRight)
{
    // 1. rotate CV::Mat at center(x,y)
    cv::Mat rotatedImage;
    if (abs(rotationRadius) < 1E-4)
    {
        // skip rotation if the rotationRadius is 0.
        rotatedImage = cv_img;
    }
    else
    {
        cv::Point2f rotation_center_pt(centerX, centerY);
        cv::Mat rotation_mat = cv::getRotationMatrix2D(rotation_center_pt, -rotationRadius * 180 / PI, 1.0);
        cv::warpAffine(cv_img, rotatedImage, rotation_mat, cv::Size(cv_img.cols, cv_img.rows));
    }

    // 2. crop CV::Mat from (x-cropLeft, y-cropUp, x+cropRight, y+cropDown)
    int x1 = (int)(centerX - cropLeft * baseWidth);
    int y1 = (int)(centerY - cropUp * baseWidth);
    int x2 = (int)(centerX + cropRight * baseWidth);
    int y2 = (int)(centerY + cropDown * baseWidth);
    x1 = std::max(x1, 0);
    x2 = std::min(x2, cv_img.cols);
    y1 = std::max(y1, 0);
    y2 = std::min(y2, cv_img.rows);
    cv::Rect crop_rect(x1, y1, x2 - x1, y2 - y1);
    // Crop the full image to that image contained by the rectangle crop_rect
    // Note that this doesn't copy the data
    cv::Mat croppedImage = rotatedImage(crop_rect);

    // 3. resize to (dstWidth, dstHeight)
    cv::Mat resizedImage;
    cv::resize(croppedImage, resizedImage, cv::Size(dstWidth, dstHeight), 0, 0, 2); // 2 for bicubic interpolation

    //cv::imshow("rotated", rotatedImage);
    //cv::imshow("cropped", croppedImage);
    //cv::imshow("resized", resizedImage);
    //cv::waitKey(0);

    // 4. CV::Mat => dst
    Datum datum;
    caffe::CVMatToDatum(resizedImage, &datum);
    dst = datum.data();
}

// crop image by rotating the image at the specified center (x,y) and crop the region at 
// (x1, y1, x2, y2), which is (centerX-cropLeft*baseWidth, centerY-cropUp*baseWidth, 
//  centerX+cropRight*baseWidth, centerY+cropDown*baseWidth) 
// The separate of baseWidth is for convenient so that the crop params can be relatively fixed for the same scenario.
#ifdef _MSC_VER
 __declspec(dllexport)
#endif
void CropImageAndResize(const unsigned char* image, int width, int height, int channels, int stride,
    std::string &dst, int dstWidth, int dstHeight,
    float centerX, float centerY, float ratationRadius,
    float baseWidth, float cropUp, float cropDown, float cropLeft, float cropRight)
{
    // src => CV::Mat
    cv::Mat cv_img = CreateCVMat(image, width, height, channels, stride);

    // crop and resize
    CropImageAndResize(cv_img, dst, dstWidth, dstHeight,
        centerX, centerY, ratationRadius,
        baseWidth, cropUp, cropDown, cropLeft, cropRight);
}

template <typename Dtype>
TsvDataLayer<Dtype>::~TsvDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void TsvDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const TsvDataParameter &tsv_param = this->layer_param().tsv_data_param();
  // open TSV file
  string tsv_data = tsv_param.source();
  bool has_shuffle_file = tsv_param.has_source_shuffle();
  string tsv_shuffle;
  if (has_shuffle_file)
      tsv_shuffle = tsv_param.source_shuffle();
  int col_data = tsv_param.col_data();
  int col_label = tsv_param.col_label();
  //int col_crop = tsv_param.col_crop();
  bool has_separate_label_file = tsv_param.has_source_label();

  tsv_.Open(tsv_data.c_str(), col_data, has_separate_label_file ? -1 : col_label);
  if (has_shuffle_file)
    tsv_.ShuffleData(tsv_shuffle);
  if (has_separate_label_file)
  {
	  string tsv_label = tsv_param.source_label();
	  tsv_label_.Open(tsv_label.c_str(), -1, col_label);
	  if (has_shuffle_file)
        tsv_label_.ShuffleData(tsv_shuffle);
	  CHECK_EQ(tsv_.TotalLines(), tsv_label_.TotalLines())
		  << "Data and label files must have the same line number: " 
		  << tsv_.TotalLines() << " vs. " << tsv_label_.TotalLines();
  }
  int batch_size = tsv_param.batch_size();
  LOG(INFO) << "Total data: " << tsv_.TotalLines() << ", Batch size: " << batch_size << ", Epoch iterations: " << (float)tsv_.TotalLines() / batch_size;

  // initialize the prefetch and top blobs.
  int new_width = tsv_param.new_width();
  int new_height = tsv_param.new_height();
  int channels = tsv_param.channels();
  int crop_size = this->layer_param().transform_param().crop_size();
  vector<int> top_shape(4);
  top_shape[0] = 1;
  top_shape[1] = channels;
  top_shape[2] = crop_size > 0 ? crop_size : new_height;
  top_shape[3] = crop_size > 0 ? crop_size : new_width;

  this->transformed_data_.Reshape(top_shape);
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
	  this->prefetch_[i].data_.Reshape(top_shape);
  }

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
	int label_dim = tsv_param.label_dim();
    vector<int> label_shape(2);
    label_shape[0] = batch_size;
    label_shape[1] = label_dim;
	top[1]->Reshape(label_shape);
	for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
	  this->prefetch_[i].label_.Reshape(label_shape);
	}
  }
}

template <typename Dtype>
void TsvDataLayer<Dtype>::process_one_image(const string &input_b64coded_data, const TsvDataParameter &tsv_param, Dtype *output_image_data)
{
    vector<BYTE> data = base64_decode(input_b64coded_data);
    if (tsv_param.data_format() == TsvDataParameter_Base64DataFormat_Image)
    {
        cv::Mat cvImg = ReadImageStreamToCVMat(data, tsv_param.new_height(), tsv_param.new_width(), tsv_param.channels() > 1);
        Datum datum;
        CVMatToDatum(cvImg, &datum);
        this->data_transformer_->TransformData(datum, output_image_data);
    }
    else if (tsv_param.data_format() == TsvDataParameter_Base64DataFormat_RawData)
    {
        caffe_copy(tsv_param.new_height() * tsv_param.new_width() * tsv_param.channels(), (Dtype*)&data[0], output_image_data);
    }
}

template <typename Dtype>
void TsvDataLayer<Dtype>::process_one_label(const string &input_label_data, const TsvDataParameter &tsv_param, Dtype *output_label_data)
{
    if (this->output_labels_) {
        int label_dim = tsv_param.label_dim();
        if (label_dim == 1)   // single label case
        {
            output_label_data[0] = atoi(input_label_data.c_str());
        }
        else
        {
            std::stringstream lineStream(input_label_data);
            string cell;
            vector<Dtype> labels;
            if (tsv_param.unroll_label())  // for multi-hot labels
            {
                labels.resize(label_dim);
                memset(&labels[0], 0, label_dim * sizeof(int));
                while (std::getline(lineStream, cell, ';'))
                {
                    int lbl = atoi(cell.c_str());
                    CHECK_LT(lbl, label_dim) << "Label value is too large! Dim = " << label_dim << ", but label is: " << lbl;
                    if (lbl >= 0)       // ignore negative padding values
                        labels[lbl] = 1;
                }
            }
            else  // for compact format labels
            {
                while (std::getline(lineStream, cell, ';'))
                {
                    labels.push_back(atoi(cell.c_str()));
                    if (labels.size() == label_dim)
                    {
                        LOG(FATAL) << "Too many labels! label_dim = " << label_dim << ", but labels are: " << input_label_data;
                        break;
                    }
                }
                for (int i = labels.size(); i < label_dim; i++)
                    labels.push_back(-1);
            }
            caffe_copy(label_dim, &labels[0], output_label_data);
        }
    }

}

// This function is called on prefetch thread
template <typename Dtype>
void TsvDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
	CPUTimer batch_timer;
	batch_timer.Start();
	double read_time = 0;
	double trans_time = 0;
	CPUTimer timer;
	CHECK(batch->data_.count());

    const TsvDataParameter &tsv_param = this->layer_param().tsv_data_param();
    
    int batch_size = tsv_param.batch_size();
    int crop_size = this->layer_param().transform_param().crop_size();

	// initialize the prefetch and top blobs.
	vector<int> top_shape(4);
    top_shape[0] = batch_size;
    top_shape[1] = tsv_param.channels();
	top_shape[2] = crop_size > 0 ? crop_size : tsv_param.new_height();
	top_shape[3] = crop_size > 0 ? crop_size : tsv_param.new_width();
	batch->data_.Reshape(top_shape);

	if (this->output_labels_) {
        memset(batch->label_.mutable_cpu_data(), 0, batch->label_.count() * sizeof(Dtype));
	}

    vector<string> base64coded_data;
    vector<string> label;
    bool has_separate_label_file = tsv_param.has_source_label();
    timer.Start();
	for (int item_id = 0; item_id < batch_size; ++item_id)
	{
		if (tsv_.ReadNextLine(base64coded_data, label) != 0)
		{
			DLOG(INFO) << "Restarting data prefetching from start.";
			tsv_.MoveToFirst();
			tsv_.ReadNextLine(base64coded_data, label);
		}
		if (has_separate_label_file)
		{
			if (tsv_label_.ReadNextLine(base64coded_data, label) != 0)
			{
				DLOG(INFO) << "Restarting label prefetching from start.";
				tsv_label_.MoveToFirst();
				tsv_label_.ReadNextLine(base64coded_data, label);
			}
		}
	}
	read_time += timer.MicroSeconds();

	timer.Start();
    #pragma omp parallel for
    for (long long i = 0.; i < base64coded_data.size(); i++)
    {
        process_one_image(base64coded_data[i], tsv_param, batch->data_.mutable_cpu_data() + batch->data_.offset(i));
        process_one_label(label[i], tsv_param, batch->label_.mutable_cpu_data() + batch->label_.offset(i));
    }
    trans_time += timer.MicroSeconds();
    
    timer.Stop();
	batch_timer.Stop();
	DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
	DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
	DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(TsvDataLayer);
REGISTER_LAYER_CLASS(TsvData);

}  // namespace caffe

