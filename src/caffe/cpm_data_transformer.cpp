#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
//#include <opencv2/opencv.hpp>
//#include <opencv2/contrib/contrib.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif  // USE_OPENCV

#include <iostream>
#include <algorithm>
#include <fstream>
//using namespace cv;
//using namespace std;

#include <string>
#include <sstream>
#include <vector>

#include "caffe/cpm_data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
//#include <omp.h>

#define BOOST_SPIRIT_THREADSAFE
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>

using boost::property_tree::ptree;
using boost::property_tree::read_json;

namespace caffe {

template<typename Dtype>
void CPMDataTransformer<Dtype>::ReadMetaData2(MetaData& meta, const std::string &data) {
	//split by ;;
	std::string data_copy = data;
	std::string delimiter = ";;";
	//first part is json string
	std::string json_data = data_copy.substr(0, data_copy.find(delimiter));
	data_copy.erase(0, data_copy.find(delimiter) + delimiter.length());
	//2nd part is mask miss
	std::string mask_miss_b64encoded = data_copy;

	vector<BYTE> mask_miss_str = base64_decode(mask_miss_b64encoded);
	meta.mask_miss = ReadImageStreamToCVMat(mask_miss_str, -1, -1, 0);

	// Read json.
	ptree pt;
	std::istringstream is(json_data);
	read_json(is, pt);

    //meta.people_index = pt.get<float>("people_index");
	//LOG(INFO) << "People index: " << meta.people_index;

        meta.img_size = cv::Size(pt.get<float>("img_width"), pt.get<float>("img_height"));
	meta.scale_self = pt.get<float>("scale_provided");
	meta.numOtherPeople = (int)pt.get<float>("numOtherPeople");
	//LOG(INFO) << "numOtherPeople: " << meta.numOtherPeople;
	vector<float> objpos;
	ptree pt_objpos = pt.get_child("objpos");   // format: x1, y1, x2, y2
	for (ptree::iterator iter = pt_objpos.begin(); iter != pt_objpos.end(); ++iter)
		objpos.push_back(iter->second.get_value<float>());
	
	meta.objpos.x = objpos[0];
	meta.objpos.y = objpos[1];
	meta.objpos -= cv::Point2f(1, 1);

	meta.joint_self.joints.resize(np_in_lmdb);
	meta.joint_self.isVisible.resize(np_in_lmdb);
	float joint_self_temp[3];
	int i = 0;
	//for (boost::property_tree::ptree::value_type &row : pt.get_child("joint_self"))
	for (ptree::iterator row = pt.get_child("joint_self").begin(); row != pt.get_child("joint_self").end(); ++row)
	{
		int j = 0;
		for (ptree::iterator cell = row->second.begin(); cell != row->second.end(); ++cell)
		//for (boost::property_tree::ptree::value_type &cell : row.second)
		{
			joint_self_temp[j] = cell->second.get_value<float>();
			//joint_self_temp[j] = cell.second.get_value<float>();
			j++;
		}
		meta.joint_self.joints[i].x = joint_self_temp[0];
		meta.joint_self.joints[i].y = joint_self_temp[1];
		meta.joint_self.joints[i] -= cv::Point2f(1, 1); //from matlab 1-index to c++ 0-index
		
		meta.joint_self.isVisible[i] = (joint_self_temp[2] == 0) ? 0 : 1;
		if (meta.joint_self.joints[i].x < 0 || meta.joint_self.joints[i].y < 0 ||
			meta.joint_self.joints[i].x >= meta.img_size.width || meta.joint_self.joints[i].y >= meta.img_size.height) {
			meta.joint_self.isVisible[i] = 2; // 2 means cropped, 0 means occluded by still on image
		}
		i++;
	}

	//others 
	meta.objpos_other.resize(meta.numOtherPeople);
	meta.joint_others.resize(meta.numOtherPeople);

	if (meta.numOtherPeople == 0) {
		//continue;
	}else if (meta.numOtherPeople == 1) {
		vector<float> objpos_other;
		ptree pt_objpos_other = pt.get_child("objpos_other");   // format: x1, y1, x2, y2
		for (ptree::iterator iter = pt_objpos_other.begin(); iter != pt_objpos_other.end(); ++iter)
			objpos_other.push_back(iter->second.get_value<float>());

		meta.objpos_other[0].x = objpos_other[0];
		meta.objpos_other[0].y = objpos_other[1];
		meta.objpos_other[0] -= cv::Point2f(1, 1);

		meta.joint_others[0].joints.resize(np_in_lmdb);
		meta.joint_others[0].isVisible.resize(np_in_lmdb);
		float joint_others_temp[3];
		int i = 0;
		for (ptree::iterator row = pt.get_child("joint_others").begin(); row != pt.get_child("joint_others").end(); ++row)
		//for (boost::property_tree::ptree::value_type &row : pt.get_child("joint_others"))
		{
			int j = 0;
			for (ptree::iterator cell = row->second.begin(); cell != row->second.end(); ++cell)
			//for (boost::property_tree::ptree::value_type &cell : row.second)
			{
				//joint_others_temp[j] = cell.second.get_value<float>();
				joint_others_temp[j] = cell->second.get_value<float>();
				j++;
			}
			meta.joint_others[0].joints[i].x = joint_others_temp[0];
			meta.joint_others[0].joints[i].y = joint_others_temp[1];
			meta.joint_others[0].joints[i] -= cv::Point2f(1, 1); //from matlab 1-index to c++ 0-index

			meta.joint_others[0].isVisible[i] = (joint_others_temp[2] == 0) ? 0 : 1;
			if (meta.joint_others[0].joints[i].x < 0 || meta.joint_others[0].joints[i].y < 0 ||
				meta.joint_others[0].joints[i].x >= meta.img_size.width || meta.joint_others[0].joints[i].y >= meta.img_size.height) {
				meta.joint_others[0].isVisible[i] = 2; // 2 means cropped, 0 means occluded by still on image
			}
			i++;
		}

	}
	else {
		float objpos_other_temp[2];
		int p = 0;
		for (ptree::iterator row = pt.get_child("objpos_other").begin(); row != pt.get_child("objpos_other").end(); ++row)
		//for (boost::property_tree::ptree::value_type &row : pt.get_child("objpos_other"))
		{
			int q = 0;
			for (ptree::iterator cell = row->second.begin(); cell != row->second.end(); ++cell)
			//for (boost::property_tree::ptree::value_type &cell : row.second)
			{
				//objpos_other_temp.push_back(cell.second.get_value<float>());
				objpos_other_temp[q] = cell->second.get_value<float>();
				//assert(objpos_other_temp[q] >= 0);
				//LOG(INFO) << "p = " << p << ", q = " << q << ", " << objpos_other_temp[q];
				q++;
			}
			//LOG(INFO) << "p = " << p;
			meta.objpos_other[p].x = objpos_other_temp[0];
			meta.objpos_other[p].y = objpos_other_temp[1];
			meta.objpos_other[p] -= cv::Point2f(1, 1);
			p++;
		}

		int k = 0;
		float joint_others_temp[3];
		for (ptree::iterator rowOut = pt.get_child("joint_others").begin(); rowOut != pt.get_child("joint_others").end(); ++rowOut) {
		//for (boost::property_tree::ptree::value_type &rowOut : pt.get_child("joint_others")) {
			meta.joint_others[k].joints.resize(np_in_lmdb);
			meta.joint_others[k].isVisible.resize(np_in_lmdb);

			int i = 0;
			for (ptree::iterator row = rowOut->second.begin(); row != rowOut->second.end(); ++row) {
			//for (boost::property_tree::ptree::value_type &row : rowOut.second) {
				int j = 0;
				for (ptree::iterator cell = row->second.begin(); cell != row->second.end(); ++cell)
				//for (boost::property_tree::ptree::value_type &cell : row.second)
				{
					joint_others_temp[j] = cell->second.get_value<float>();
					j++;
				}
				meta.joint_others[k].joints[i].x = joint_others_temp[0];
				meta.joint_others[k].joints[i].y = joint_others_temp[1];
				meta.joint_others[k].joints[i] -= cv::Point2f(1, 1); //from matlab 1-index to c++ 0-index

				meta.joint_others[k].isVisible[i] = (joint_others_temp[2] == 0) ? 0 : 1;
				if (meta.joint_others[k].joints[i].x < 0 || meta.joint_others[k].joints[i].y < 0 ||
					meta.joint_others[k].joints[i].x >= meta.img_size.width || meta.joint_others[k].joints[i].y >= meta.img_size.height) {
					meta.joint_others[k].isVisible[i] = 2; // 2 means cropped, 0 means occluded by still on image
				}
				i++;
			}
			k++;
		}
	}
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::TransformMetaJoints(MetaData& meta) {
  //transform joints in meta from np_in_lmdb (specified in prototxt) to np (specified in prototxt)
  TransformJoints(meta.joint_self);
  for(int i=0;i<meta.joint_others.size();i++){
    TransformJoints(meta.joint_others[i]);
  }
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::TransformJoints(Joints& j) {
  //transform joints in meta from np_in_lmdb (specified in prototxt) to np (specified in prototxt)
  Joints jo = j;

  if(np == 56){
    int COCO_to_ours_1[18] = {1,6, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
    int COCO_to_ours_2[18] = {1,7, 7,9,11, 6,8,10, 13,15,17, 12,14,16, 3,2,5,4};
    jo.joints.resize(np);
    jo.isVisible.resize(np);
    for(int i=0;i<18;i++){
      jo.joints[i] = (j.joints[COCO_to_ours_1[i]-1] + j.joints[COCO_to_ours_2[i]-1]) * 0.5;
      if(j.isVisible[COCO_to_ours_1[i]-1]==2 || j.isVisible[COCO_to_ours_2[i]-1]==2){
        jo.isVisible[i] = 2;
      }
      else if(j.isVisible[COCO_to_ours_1[i]-1]==3 || j.isVisible[COCO_to_ours_2[i]-1]==3){
        jo.isVisible[i] = 3;
      }
      else {
        jo.isVisible[i] = j.isVisible[COCO_to_ours_1[i]-1] && j.isVisible[COCO_to_ours_2[i]-1];
      }
    }
  }

  else if(np == 43){
    int MPI_to_ours_1[15] = {9, 8,12,11,10,13,14,15, 2, 1, 0, 3, 4, 5, 7};
    int MPI_to_ours_2[15] = {9, 8,12,11,10,13,14,15, 2, 1, 0, 3, 4, 5, 6};
    jo.joints.resize(np);
    jo.isVisible.resize(np);

    for(int i=0;i<15;i++){
      jo.joints[i] = (j.joints[MPI_to_ours_1[i]] + j.joints[MPI_to_ours_2[i]]) * 0.5;
      if(j.isVisible[MPI_to_ours_1[i]]==2 || j.isVisible[MPI_to_ours_2[i]]==2){
        jo.isVisible[i] = 2;
      }
      else {
        jo.isVisible[i] = j.isVisible[MPI_to_ours_1[i]] && j.isVisible[MPI_to_ours_2[i]];
      }
    }
  }

  j = jo;
}

template<typename Dtype> CPMDataTransformer<Dtype>::CPMDataTransformer(const CPMTransformationParameter& param, Phase phase) : param_(param), phase_(phase) {
  // check if we want to use mean_file
  if (param_.has_mean_file()) {
    CHECK_EQ(param_.mean_value_size(), 0) <<
      "Cannot specify mean_file and mean_value at the same time";
    const std::string& mean_file = param.mean_file();
    if (Caffe::root_solver()) {
      LOG(INFO) << "Loading mean file from: " << mean_file;
    }
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) {
    CHECK(param_.has_mean_file() == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    for (int c = 0; c < param_.mean_value_size(); ++c) {
      mean_values_.push_back(param_.mean_value(c));
    }
  }
  LOG(INFO) << "CPMDataTransformer constructor done.";
  np_in_lmdb = param_.np_in_lmdb();
  np = param_.num_parts();
  is_table_set = false;
}

template<typename Dtype> cv::Mat CPMDataTransformer<Dtype>::ReadImageStreamToCVMat(vector<unsigned char>& imbuf, const int height, const int width, const bool is_color)
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

template<typename Dtype> void CPMDataTransformer<Dtype>::Transform_nv2(const std::string &input_b64coded_data, const std::string &input_label_data, Dtype* transformed_data, Dtype* transformed_label) {

	CPUTimer timer0;
	timer0.Start();

	CPUTimer timer1;
	timer1.Start();

	vector<BYTE> data = base64_decode(input_b64coded_data);
	cv::Mat img = ReadImageStreamToCVMat(data, -1, -1, 1);
	
	MetaData meta;
	ReadMetaData2(meta, input_label_data);

	//TODO: some parameter should be set in prototxt
	int clahe_tileSize = param_.clahe_tile_size();
	int clahe_clipLimit = param_.clahe_clip_limit();
	AugmentSelection as = {
		false,
		0.0,
		cv::Size(),
		0,
	};

	const int mode = 5;
	int crop_x = param_.crop_size_x();
	int crop_y = param_.crop_size_y();
	cv::Mat mask_miss;
	mask_miss = meta.mask_miss;

	VLOG(2) << "  base64 decoding: " << timer1.MicroSeconds() / 1000.0 << " ms";
	timer1.Start();

	//color, contract
	if (param_.do_clahe())
		clahe(img, clahe_tileSize, clahe_clipLimit);
	if (param_.gray() == 1) {
		cv::cvtColor(img, img, CV_BGR2GRAY);
		cv::cvtColor(img, img, CV_GRAY2BGR);
	}
	VLOG(2) << "  color: " << timer1.MicroSeconds() / 1000.0 << " ms";
	timer1.Start();

	int stride = param_.stride();
	if (param_.transform_body_joint()) // we expect to transform body joints, and not to transform hand joints
		TransformMetaJoints(meta);

	VLOG(2) << "  ReadMeta+MetaJoints: " << timer1.MicroSeconds() / 1000.0 << " ms";
	timer1.Start();

	//Start transforming
	cv::Mat img_aug = cv::Mat::zeros(crop_y, crop_x, CV_8UC3);
	cv::Mat mask_miss_aug;
	cv::Mat img_temp, img_temp2, img_temp3; //size determined by scale
	VLOG(2) << "   input size (" << img.cols << ", " << img.rows << ")";
	// We only do random transform as augmentation when training.
	//if (phase_ == TRAIN) {
		as.scale = augmentation_scale(img, img_temp, mask_miss, meta, mode);
		as.degree = augmentation_rotate(img_temp, img_temp2, mask_miss, meta, mode);
		as.crop = augmentation_croppad(img_temp2, img_temp3, mask_miss, mask_miss_aug, meta, mode);
		as.flip = augmentation_flip(img_temp3, img_aug, mask_miss_aug, meta, mode);
	//}
	//else {
	//	img_aug = img.clone();
	//	as.scale = 1;
	//	as.crop = cv::Size();
	//	as.flip = 0;
	//	as.degree = 0;
	//	mask_miss_aug = mask_miss.clone();
	//}
	cv::resize(mask_miss_aug, mask_miss_aug, cv::Size(), 1.0 / stride, 1.0 / stride, cv::INTER_CUBIC);

	VLOG(2) << "  Aug: " << timer1.MicroSeconds() / 1000.0 << " ms";
	timer1.Start();
	//LOG(INFO) << "scale: " << as.scale << "; crop:(" << as.crop.width << "," << as.crop.height 
	//          << "); flip:" << as.flip << "; degree: " << as.degree;

	//copy transformed img (img_aug) into transformed_data, do the mean-subtraction here
	int offset = img_aug.rows * img_aug.cols;
	int rezX = img_aug.cols;
	int rezY = img_aug.rows;
	int grid_x = rezX / stride;
	int grid_y = rezY / stride;
	int channelOffset = grid_y * grid_x;

	for (int i = 0; i < img_aug.rows; ++i) {
		for (int j = 0; j < img_aug.cols; ++j) {
			cv::Vec3b& rgb = img_aug.at<cv::Vec3b>(i, j);
			transformed_data[0 * offset + i*img_aug.cols + j] = (rgb[0] - 128) / 256.0;
			transformed_data[1 * offset + i*img_aug.cols + j] = (rgb[1] - 128) / 256.0;
			transformed_data[2 * offset + i*img_aug.cols + j] = (rgb[2] - 128) / 256.0;
		}
	}

	// label size is image size/ stride
	if (mode > 4) {
		for (int g_y = 0; g_y < grid_y; g_y++) {
			for (int g_x = 0; g_x < grid_x; g_x++) {
				for (int i = 0; i < np; i++) {
					float weight = float(mask_miss_aug.at<uchar>(g_y, g_x)) / 255; //mask_miss_aug.at<uchar>(i, j); 
																				   // VLOG(2) << "i = " << i << ", isVisible = " << meta.joint_self.isVisible[i];
					if (meta.joint_self.isVisible[i] != 3) {
						transformed_label[i*channelOffset + g_y*grid_x + g_x] = weight;
					}
				}
				// background channel
				if (mode == 5) {
					transformed_label[np*channelOffset + g_y*grid_x + g_x] = float(mask_miss_aug.at<uchar>(g_y, g_x)) / 255;
				}
			}
		}
	}

	//putGaussianMaps(transformed_data + 3*offset, meta.objpos, 1, img_aug.cols, img_aug.rows, param_.sigma_center());
	//LOG(INFO) << "image transformation done!";
	generateLabelMap(transformed_label, img_aug, meta);

	VLOG(2) << "  putGauss+genLabel: " << timer1.MicroSeconds() / 1000.0 << " ms";

	VLOG(2) << "  Inside Transform_nv: " << timer0.MicroSeconds() / 1000.0 << " ms";
}
// include mask_miss
template<typename Dtype>
float CPMDataTransformer<Dtype>::augmentation_scale(cv::Mat& img_src, cv::Mat& img_temp, cv::Mat& mask_miss, MetaData& meta, int mode) {
	float scale_multiplier;
	if (phase_ == TRAIN) {
		float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
		//float dice = 1.0;
		//float scale_multiplier;
		//float scale = (param_.scale_max() - param_.scale_min()) * dice + param_.scale_min(); //linear shear into [scale_min, scale_max]
		if (dice > param_.scale_prob()) {
			img_temp = img_src.clone();
			scale_multiplier = 1;
		}
		else {
			float dice2 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
			//float dice2 = 0.0;
			scale_multiplier = (param_.scale_max() - param_.scale_min()) * dice2 + param_.scale_min(); //linear shear into [scale_min, scale_max]
		}
	}
	else {
		scale_multiplier = 1.0;
	}
  float scale_abs = param_.target_dist()/meta.scale_self;
  float scale = scale_abs * scale_multiplier;
  cv::resize(img_src, img_temp, cv::Size(), scale, scale, cv::INTER_CUBIC);
  if(mode>4){
    cv::resize(mask_miss, mask_miss, cv::Size(), scale, scale, cv::INTER_CUBIC);
  }

  //modify meta data
  meta.objpos *= scale;
  for(int i=0; i<np; i++){
    meta.joint_self.joints[i] *= scale;
  }
  for(int p=0; p<meta.numOtherPeople; p++){
    meta.objpos_other[p] *= scale;
    for(int i=0; i<np; i++){
      meta.joint_others[p].joints[i] *= scale;
    }
  }
  return scale_multiplier;
}

template<typename Dtype>
cv::Size CPMDataTransformer<Dtype>::augmentation_croppad(cv::Mat& img_src, cv::Mat& img_dst, cv::Mat& mask_miss, cv::Mat& mask_miss_aug, MetaData& meta, int mode) {
	float dice_x;
	float dice_y;
	if (phase_ == TRAIN) {
		dice_x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
		dice_y = static_cast <float> (rand()) / static_cast <float> (RAND_MAX); //[0,1]
		//float dice_x = 0.0;
		//float dice_y = 1.0;
	}
	else {
		dice_x = 0.5;
		dice_y = 0.5;
	}
  int crop_x = param_.crop_size_x();
  int crop_y = param_.crop_size_y();

  float x_offset = int((dice_x - 0.5) * 2 * param_.center_perterb_max());
  float y_offset = int((dice_y - 0.5) * 2 * param_.center_perterb_max());

  //LOG(INFO) << "cv::Size of img_temp is " << img_temp.cols << " " << img_temp.rows;
  //LOG(INFO) << "ROI is " << x_offset << " " << y_offset << " " << min(800, img_temp.cols) << " " << min(256, img_temp.rows);
  cv::Point2i center = meta.objpos + cv::Point2f(x_offset, y_offset);
  int offset_left = -(center.x - (crop_x/2));
  int offset_up = -(center.y - (crop_y/2));
  // int to_pad_right = std::max(center.x + (crop_x - crop_x/2) - img_src.cols, 0);
  // int to_pad_down = std::max(center.y + (crop_y - crop_y/2) - img_src.rows, 0);
  
  img_dst = cv::Mat::zeros(crop_y, crop_x, CV_8UC3) + cv::Scalar(128,128,128);
  mask_miss_aug = cv::Mat::zeros(crop_y, crop_x, CV_8UC1) + cv::Scalar(255); //for MPI, COCO with cv::Scalar(255);
  for(int i=0;i<crop_y;i++){
    for(int j=0;j<crop_x;j++){ //i,j on cropped
      int coord_x_on_img = center.x - crop_x/2 + j;
      int coord_y_on_img = center.y - crop_y/2 + i;
      if(onPlane(cv::Point(coord_x_on_img, coord_y_on_img), cv::Size(img_src.cols, img_src.rows))){
        img_dst.at<cv::Vec3b>(i,j) = img_src.at<cv::Vec3b>(coord_y_on_img, coord_x_on_img);
        if(mode>4){
          mask_miss_aug.at<uchar>(i,j) = mask_miss.at<uchar>(coord_y_on_img, coord_x_on_img);
        }
      }
    }
  }

  //modify meta data
  cv::Point2f offset(offset_left, offset_up);
  meta.objpos += offset;
  for(int i=0; i<np; i++){
    meta.joint_self.joints[i] += offset;
  }
  for(int p=0; p<meta.numOtherPeople; p++){
    meta.objpos_other[p] += offset;
    for(int i=0; i<np; i++){
      meta.joint_others[p].joints[i] += offset;
    }
  }

  return cv::Size(x_offset, y_offset);
}

template<typename Dtype>
bool CPMDataTransformer<Dtype>::augmentation_flip(cv::Mat& img_src, cv::Mat& img_aug, cv::Mat& mask_miss, MetaData& meta, int mode) {
  bool doflip;
  if (phase_ == TRAIN) {
	  if (param_.aug_way() == "rand") {
		  float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		  doflip = (dice <= param_.flip_prob());
		  //doflip = true;
	  }
	  else if (param_.aug_way() == "table") {
		  doflip = (aug_flips[meta.write_number][meta.epoch % param_.num_total_augs()] == 1);
	  }
	  else {
		  doflip = 0;
		  LOG(INFO) << "Unhandled exception!!!!!!";
	  }
  }
  else {
	  doflip = 0;
  }
  if(doflip){
    flip(img_src, img_aug, 1);
    int w = img_src.cols;
    if(mode>4){
      flip(mask_miss, mask_miss, 1);
    }
    meta.objpos.x = w - 1 - meta.objpos.x;
    for(int i=0; i<np; i++){
      meta.joint_self.joints[i].x = w - 1 - meta.joint_self.joints[i].x;
    }
    if(param_.transform_body_joint())
      swapLeftRight(meta.joint_self);

    for(int p=0; p<meta.numOtherPeople; p++){
      meta.objpos_other[p].x = w - 1 - meta.objpos_other[p].x;
      for(int i=0; i<np; i++){
        meta.joint_others[p].joints[i].x = w - 1 - meta.joint_others[p].joints[i].x;
      }
      if(param_.transform_body_joint())
        swapLeftRight(meta.joint_others[p]);
    }
  }
  else {
    img_aug = img_src.clone();
  }
  return doflip;
}

template<typename Dtype>
float CPMDataTransformer<Dtype>::augmentation_rotate(cv::Mat& img_src, cv::Mat& img_dst, cv::Mat& mask_miss, MetaData& meta, int mode) {
  
  float degree;
  if (phase_ == TRAIN) {
	  if (param_.aug_way() == "rand") {
		  float dice = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		  //float dice = 0.8;
		  degree = (dice - 0.5) * 2 * param_.max_rotate_degree();
	  }
	  else if (param_.aug_way() == "table") {
		  degree = aug_degs[meta.write_number][meta.epoch % param_.num_total_augs()];
	  }
	  else {
		  degree = 0;
		  LOG(INFO) << "Unhandled exception!!!!!!";
	  }
  }
  else {
	  degree = 0;
  }
  cv::Point2f center(img_src.cols/2.0, img_src.rows/2.0);
  cv::Mat R = getRotationMatrix2D(center, degree, 1.0);
  cv::Rect bbox = cv::RotatedRect(center, img_src.size(), degree).boundingRect();
  // adjust transformation matrix
  R.at<double>(0,2) += bbox.width/2.0 - center.x;
  R.at<double>(1,2) += bbox.height/2.0 - center.y;
  //LOG(INFO) << "R=[" << R.at<double>(0,0) << " " << R.at<double>(0,1) << " " << R.at<double>(0,2) << ";" 
  //          << R.at<double>(1,0) << " " << R.at<double>(1,1) << " " << R.at<double>(1,2) << "]";
  warpAffine(img_src, img_dst, R, bbox.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(128,128,128));
  if(mode >4){
    warpAffine(mask_miss, mask_miss, R, bbox.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(255)); //cv::Scalar(0) for MPI, COCO with cv::Scalar(255);
  }

  //adjust meta data
  RotatePoint(meta.objpos, R);
  for(int i=0; i<np; i++){
    RotatePoint(meta.joint_self.joints[i], R);
  }
  for(int p=0; p<meta.numOtherPeople; p++){
    RotatePoint(meta.objpos_other[p], R);
    for(int i=0; i<np; i++){
      RotatePoint(meta.joint_others[p].joints[i], R);
    }
  }
  return degree;
}
// end here

template<typename Dtype>
bool CPMDataTransformer<Dtype>::onPlane(cv::Point p, cv::Size img_size) {
  if(p.x < 0 || p.y < 0) return false;
  if(p.x >= img_size.width || p.y >= img_size.height) return false;
  return true;
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::swapLeftRight(Joints& j) {
  if(np == 56){
    int right[8] = {3,4,5, 9,10,11,15,17}; 
    int left[8] =  {6,7,8,12,13,14,16,18}; 
    for(int i=0; i<8; i++){    
      int ri = right[i] - 1;
      int li = left[i] - 1;
      cv::Point2f temp = j.joints[ri];
      j.joints[ri] = j.joints[li];
      j.joints[li] = temp;
      int temp_v = j.isVisible[ri];
      j.isVisible[ri] = j.isVisible[li];
      j.isVisible[li] = temp_v;
    }
  }

  else if(np == 43){
    int right[6] = {3,4,5,9,10,11}; 
    int left[6] = {6,7,8,12,13,14}; 
    for(int i=0; i<6; i++){   
      int ri = right[i] - 1;
      int li = left[i] - 1;
      cv::Point2f temp = j.joints[ri];
      j.joints[ri] = j.joints[li];
      j.joints[li] = temp;
      int temp_v = j.isVisible[ri];
      j.isVisible[ri] = j.isVisible[li];
      j.isVisible[li] = temp_v;
    }
  }
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::RotatePoint(cv::Point2f& p, cv::Mat R){
  cv::Mat point(3,1,CV_64FC1);
  point.at<double>(0,0) = p.x;
  point.at<double>(1,0) = p.y;
  point.at<double>(2,0) = 1;
  cv::Mat new_point = R * point;
  p.x = new_point.at<double>(0,0);
  p.y = new_point.at<double>(1,0);
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::putGaussianMaps(Dtype* entry, cv::Point2f center, int stride, int grid_x, int grid_y, float sigma){
  //LOG(INFO) << "putGaussianMaps here we start for " << center.x << " " << center.y;
  float start = stride/2.0 - 0.5; //0 if stride = 1, 0.5 if stride = 2, 1.5 if stride = 4, ...
  for (int g_y = 0; g_y < grid_y; g_y++){
    for (int g_x = 0; g_x < grid_x; g_x++){
      float x = start + g_x * stride;
      float y = start + g_y * stride;
      float d2 = (x-center.x)*(x-center.x) + (y-center.y)*(y-center.y);
      float exponent = d2 / 2.0 / sigma / sigma;
      if(exponent > 4.6052){ //ln(100) = -ln(1%)
        continue;
      }
      entry[g_y*grid_x + g_x] += exp(-exponent);
      if(entry[g_y*grid_x + g_x] > 1) 
        entry[g_y*grid_x + g_x] = 1;
    }
  }
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::putVecMaps(Dtype* entryX, Dtype* entryY, cv::Mat& count, cv::Point2f centerA, cv::Point2f centerB, int stride, int grid_x, int grid_y, float sigma, int thre){
  //int thre = 4;
  centerB = centerB*0.125;
  centerA = centerA*0.125;
  cv::Point2f bc = centerB - centerA;
  int min_x = std::max( int(round(std::min(centerA.x, centerB.x)-thre)), 0);
  int max_x = std::min( int(round(std::max(centerA.x, centerB.x)+thre)), grid_x);

  int min_y = std::max( int(round(std::min(centerA.y, centerB.y)-thre)), 0);
  int max_y = std::min( int(round(std::max(centerA.y, centerB.y)+thre)), grid_y);

  float norm_bc = sqrt(bc.x*bc.x + bc.y*bc.y);
  bc.x = bc.x /norm_bc;
  bc.y = bc.y /norm_bc;

  // float x_p = (centerA.x + centerB.x) / 2;
  // float y_p = (centerA.y + centerB.y) / 2;
  // float angle = atan2f(centerB.y - centerA.y, centerB.x - centerA.x);
  // float sine = sinf(angle);
  // float cosine = cosf(angle);
  // float a_sqrt = (centerA.x - x_p) * (centerA.x - x_p) + (centerA.y - y_p) * (centerA.y - y_p);
  // float b_sqrt = 10; //fixed

  for (int g_y = min_y; g_y < max_y; g_y++){
    for (int g_x = min_x; g_x < max_x; g_x++){
      cv::Point2f ba;
      ba.x = g_x - centerA.x;
      ba.y = g_y - centerA.y;
      float dist = std::abs(ba.x*bc.y -ba.y*bc.x);

      // float A = cosine * (g_x - x_p) + sine * (g_y - y_p);
      // float B = sine * (g_x - x_p) - cosine * (g_y - y_p);
      // float judge = A * A / a_sqrt + B * B / b_sqrt;

      if(dist <= thre){
      //if(judge <= 1){
        int cnt = count.at<uchar>(g_y, g_x);
        //LOG(INFO) << "putVecMaps here we start for " << g_x << " " << g_y;
        if (cnt == 0){
          entryX[g_y*grid_x + g_x] = bc.x;
          entryY[g_y*grid_x + g_x] = bc.y;
        }
        else{
          entryX[g_y*grid_x + g_x] = (entryX[g_y*grid_x + g_x]*cnt + bc.x) / (cnt + 1);
          entryY[g_y*grid_x + g_x] = (entryY[g_y*grid_x + g_x]*cnt + bc.y) / (cnt + 1);
          count.at<uchar>(g_y, g_x) = cnt + 1;
        }
      }

    }
  }
}

template<typename Dtype>
void CPMDataTransformer<Dtype>::generateLabelMap(Dtype* transformed_label, cv::Mat& img_aug, MetaData meta) {
  int rezX = img_aug.cols;
  int rezY = img_aug.rows;
  int stride = param_.stride();
  int grid_x = rezX / stride;
  int grid_y = rezY / stride;
  int channelOffset = grid_y * grid_x;
  int mode = 5; // TO DO: make this as a parameter

  for (int g_y = 0; g_y < grid_y; g_y++){
    for (int g_x = 0; g_x < grid_x; g_x++){
      for (int i = np+1; i < 2*(np+1); i++){
        if (mode == 6 && i == (2*np + 1))
          continue;
        transformed_label[i*channelOffset + g_y*grid_x + g_x] = 0;
      }
    }
  }

  if (np == 56){
    for (int i = 0; i < 18; i++){
      cv::Point2f center = meta.joint_self.joints[i];
      if(meta.joint_self.isVisible[i] <= 1){
        putGaussianMaps(transformed_label + (i+np+39)*channelOffset, center, param_.stride(), 
                        grid_x, grid_y, param_.sigma()); //self
      }
      for(int j = 0; j < meta.numOtherPeople; j++){ //for every other person
        cv::Point2f center = meta.joint_others[j].joints[i];
        if(meta.joint_others[j].isVisible[i] <= 1){
          putGaussianMaps(transformed_label + (i+np+39)*channelOffset, center, param_.stride(), 
                          grid_x, grid_y, param_.sigma());
        }
      }
    }

    int mid_1[19] = {2, 9,  10, 2,  12, 13, 2, 3, 4, 3,  2, 6, 7, 6,  2, 1,  1,  15, 16};
    int mid_2[19] = {9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18};
    int thre = 1;

    for(int i=0;i<19;i++){
      cv::Mat count = cv::Mat::zeros(grid_y, grid_x, CV_8UC1);
      Joints jo = meta.joint_self;
      if(jo.isVisible[mid_1[i]-1]<=1 && jo.isVisible[mid_2[i]-1]<=1){
        //putVecPeaks
        putVecMaps(transformed_label + (np+ 1+ 2*i)*channelOffset, transformed_label + (np+ 2+ 2*i)*channelOffset, 
                  count, jo.joints[mid_1[i]-1], jo.joints[mid_2[i]-1], param_.stride(), grid_x, grid_y, param_.sigma(), thre); //self
      }

      for(int j = 0; j < meta.numOtherPeople; j++){ //for every other person
        Joints jo2 = meta.joint_others[j];
        if(jo2.isVisible[mid_1[i]-1]<=1 && jo2.isVisible[mid_2[i]-1]<=1){
          //putVecPeaks
          putVecMaps(transformed_label + (np+ 1+ 2*i)*channelOffset, transformed_label + (np+ 2+ 2*i)*channelOffset, 
                  count, jo2.joints[mid_1[i]-1], jo2.joints[mid_2[i]-1], param_.stride(), grid_x, grid_y, param_.sigma(), thre); //self
        }
      }
    }

    //put background channel
    for (int g_y = 0; g_y < grid_y; g_y++){
      for (int g_x = 0; g_x < grid_x; g_x++){
        float maximum = 0;
        //second background channel
        for (int i = np+39; i < np+57; i++){
          maximum = (maximum > transformed_label[i*channelOffset + g_y*grid_x + g_x]) ? maximum : transformed_label[i*channelOffset + g_y*grid_x + g_x];
        }
        transformed_label[(2*np+1)*channelOffset + g_y*grid_x + g_x] = std::max(1.0-maximum, 0.0);
      }
    }
    //LOG(INFO) << "background put";
  }
  
  else if (np == 43){
    for (int i = 0; i < 15; i++){
      cv::Point2f center = meta.joint_self.joints[i];
      if(meta.joint_self.isVisible[i] <= 1){
        putGaussianMaps(transformed_label + (i+np+29)*channelOffset, center, param_.stride(), 
                        grid_x, grid_y, param_.sigma()); //self
      }
      for(int j = 0; j < meta.numOtherPeople; j++){ //for every other person
        cv::Point2f center = meta.joint_others[j].joints[i];
        if(meta.joint_others[j].isVisible[i] <= 1){
          putGaussianMaps(transformed_label + (i+np+29)*channelOffset, center, param_.stride(), 
                          grid_x, grid_y, param_.sigma());
        }
      }
    }

    int mid_1[14] = {0, 1, 2, 3, 1, 5, 6, 1, 14, 8, 9,  14, 11, 12};
    int mid_2[14] = {1, 2, 3, 4, 5, 6, 7, 14, 8, 9, 10, 11, 12, 13};
    int thre = 1;

    for(int i=0;i<14;i++){
      cv::Mat count = cv::Mat::zeros(grid_y, grid_x, CV_8UC1);
      Joints jo = meta.joint_self;
      if(jo.isVisible[mid_1[i]]<=1 && jo.isVisible[mid_2[i]]<=1){
        //putVecPeaks
        putVecMaps(transformed_label + (np+ 1+ 2*i)*channelOffset, transformed_label + (np+ 2+ 2*i)*channelOffset, 
                  count, jo.joints[mid_1[i]], jo.joints[mid_2[i]], param_.stride(), grid_x, grid_y, param_.sigma(), thre); //self
      }

      for(int j = 0; j < meta.numOtherPeople; j++){ //for every other person
        Joints jo2 = meta.joint_others[j];
        if(jo2.isVisible[mid_1[i]]<=1 && jo2.isVisible[mid_2[i]]<=1){
          //putVecPeaks
          putVecMaps(transformed_label + (np+ 1+ 2*i)*channelOffset, transformed_label + (np+ 2+ 2*i)*channelOffset, 
                  count, jo2.joints[mid_1[i]], jo2.joints[mid_2[i]], param_.stride(), grid_x, grid_y, param_.sigma(), thre); //self
        }
      }
    }

    //put background channel
    for (int g_y = 0; g_y < grid_y; g_y++){
      for (int g_x = 0; g_x < grid_x; g_x++){
        float maximum = 0;
        //second background channel
        for (int i = np+29; i < np+44; i++){
          maximum = (maximum > transformed_label[i*channelOffset + g_y*grid_x + g_x]) ? maximum : transformed_label[i*channelOffset + g_y*grid_x + g_x];
        }
        transformed_label[(2*np+1)*channelOffset + g_y*grid_x + g_x] = std::max(1.0-maximum, 0.0);
      }
    }
    //LOG(INFO) << "background put";
  }

}

template <typename Dtype>
void CPMDataTransformer<Dtype>::InitRand() {
  const bool needs_rand = param_.mirror() ||
      (phase_ == TRAIN && param_.crop_size());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
int CPMDataTransformer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

template <typename Dtype>
void CPMDataTransformer<Dtype>::clahe(cv::Mat& bgr_image, int tileSize, int clipLimit) {
  cv::Mat lab_image;
  cv::cvtColor(bgr_image, lab_image, CV_BGR2Lab);

  // Extract the L channel
  vector<cv::Mat> lab_planes(3);
  cv::split(lab_image, lab_planes);  // now we have the L image in lab_planes[0]

  // apply the CLAHE algorithm to the L channel
  cv::Ptr<cv::CLAHE> clahe = createCLAHE(clipLimit, cv::Size(tileSize, tileSize));
  //clahe->setClipLimit(4);
  cv::Mat dst;
  clahe->apply(lab_planes[0], dst);

  // Merge the the color planes back into an Lab image
  dst.copyTo(lab_planes[0]);
  cv::merge(lab_planes, lab_image);

  // convert back to RGB
  cv::Mat image_clahe;
  cv::cvtColor(lab_image, image_clahe, CV_Lab2BGR);
  bgr_image = image_clahe.clone();
}

INSTANTIATE_CLASS(CPMDataTransformer);

}  // namespace caffe
