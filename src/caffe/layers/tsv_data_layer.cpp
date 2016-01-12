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
  int col_data = tsv_param.col_data();
  int col_label = tsv_param.col_label();
  //int col_crop = tsv_param.col_crop();
  bool has_separate_label_file = tsv_param.has_source_label();

  tsv_.Open(tsv_data.c_str(), col_data, has_separate_label_file ? -1 : col_label);
  tsv_.ShuffleData();
  if (has_separate_label_file)
  {
	  string tsv_label = tsv_param.source_label();
	  tsv_label_.Open(tsv_label.c_str(), -1, col_label);
	  tsv_label_.ShuffleData();
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
class thread_closure
{
public:
	thread_closure(vector<string>& b64data, vector<string>& rlabel):
		base64coded_data(b64data), label(rlabel)
	{ 
	}
	Batch<Dtype>* batch;
	int batch_size;
	int new_width, new_height;
	int channels;
    int label_dim;
    bool unroll_label;
    TsvDataParameter::Base64DataFormat data_format;
	vector<string>& base64coded_data;
	vector<string>& label;
	Dtype *top_data;
	Dtype *top_label;
};
INSTANTIATE_CLASS(thread_closure);

template <typename Dtype>
void TsvDataLayer<Dtype>::transform_datum(thread_closure<Dtype>& c, size_t dst_index)
{
	int i = dst_index;

    int offset = c.batch->data_.offset(i);
    vector<BYTE> data = base64_decode(c.base64coded_data[i]);
    if (c.data_format == TsvDataParameter_Base64DataFormat_Image)
    {
        cv::Mat cvImg = ReadImageStreamToCVMat(data, c.new_height, c.new_width, c.channels > 1);
        Datum datum;
        CVMatToDatum(cvImg, &datum);
        this->transformed_data_.set_cpu_data(c.top_data + offset);
        this->data_transformer_->Transform(datum, &(this->transformed_data_));
    }
    else if (c.data_format == TsvDataParameter_Base64DataFormat_RawData)
    {
        caffe_copy(c.new_height * c.new_width * c.channels, (Dtype*)&data[0], c.top_data + offset);
    }
	// Copy label.
	if (this->output_labels_) {
        if (c.label_dim == 1)   // single label case
        {
          c.top_label[i] = atoi(c.label[i].c_str());
        }
        else
        {
            int label_offset = c.batch->label_.offset(i);
            std::stringstream lineStream(c.label[i]);
            string cell;
            vector<Dtype> labels;
            if (c.unroll_label)  // for multi-hot labels
            {
                labels.resize(c.label_dim);
                memset(&labels[0], 0, c.label_dim * sizeof(int));
                while (std::getline(lineStream, cell, ';'))
                {
                    int lbl = atoi(cell.c_str());
                    CHECK_LT(lbl, c.label_dim) << "Label value is too large! Dim = " << c.label_dim << ", but label is: " << lbl;
                    if (lbl >= 0)       // ignore negative padding values
                        labels[lbl] = 1;
                }
            }
            else  // for compact format labels
            {
                while (std::getline(lineStream, cell, ';'))
                {
                    labels.push_back(atoi(cell.c_str()));
                    if (labels.size() == c.label_dim)
                    {
                        LOG(FATAL) << "Too many labels! label_dim = " << c.label_dim << ", but labels are: " << c.label[i];
                        break;
                    }
                }
                for (int i = labels.size(); i < c.label_dim; i++)
                    labels.push_back(-1);
            }
            caffe_copy(c.label_dim, &labels[0], c.top_label + label_offset);
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
	CHECK(this->transformed_data_.count());

    const TsvDataParameter &tsv_param = this->layer_param().tsv_data_param();
    
    vector<string> base64coded_data;
	vector<string> label;
	thread_closure<Dtype> c(base64coded_data, label);
	c.batch = batch;
	c.batch_size = tsv_param.batch_size();
	c.new_height = tsv_param.new_height();
	c.new_width = tsv_param.new_width();
	c.channels = tsv_param.channels();
    c.label_dim = tsv_param.label_dim();
    c.unroll_label = tsv_param.unroll_label();
    c.data_format = tsv_param.data_format();
	int crop_size = this->layer_param().transform_param().crop_size();

	// initialize the prefetch and top blobs.
	vector<int> top_shape(4);
	top_shape[0] = 1;
	top_shape[1] = c.channels;
	top_shape[2] = crop_size > 0 ? crop_size : c.new_height;
	top_shape[3] = crop_size > 0 ? crop_size : c.new_width;
	this->transformed_data_.Reshape(top_shape);
	top_shape[0] = c.batch_size;
	batch->data_.Reshape(top_shape);

	c.top_data = batch->data_.mutable_cpu_data();
	c.top_label = NULL;  // suppress warnings about uninitialized variables

	if (this->output_labels_) {
		c.top_label = batch->label_.mutable_cpu_data();
        memset(c.top_label, 0, batch->label_.count() * sizeof(Dtype));
	}

    bool has_separate_label_file = tsv_param.has_source_label();
    timer.Start();
	for (int item_id = 0; item_id < c.batch_size; ++item_id)
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
	// The following commented block is replaced by using boost::thread for portability to both Windows and Linux.
	//parallel_for((size_t)0, base64coded_data.size(), [&](size_t i){
	//	//int i = 0;
	//	vector<BYTE> img = base64_decode(base64coded_data[i]);
	//	cv::Mat cvImg = ReadImageStreamToCVMat(img, c.new_height, c.new_width, c.is_color);
	//	Datum datum;
	//	CVMatToDatum(cvImg, &datum);
	//	int offset = batch->data_.offset(i);
	//	this->transformed_data_.set_cpu_data(c.top_data + offset);
	//	this->data_transformer_->Transform(datum, &(this->transformed_data_));
	//	// Copy label.
	//	if (this->output_labels_) {
	//		c.top_label[i] = label[i];
	//	}
	//}
	//);
	boost::thread_group threads;
	for (size_t i = 0.; i < base64coded_data.size(); i++)
	{
		threads.create_thread(boost::bind(&TsvDataLayer<Dtype>::transform_datum, this, boost::ref(c), i));
	}
	threads.join_all();

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

