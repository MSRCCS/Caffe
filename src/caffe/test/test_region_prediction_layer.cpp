#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"

#include "caffe/layers/region_loss_layer.hpp"
#include "caffe/layers/region_prediction_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {
#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class RegionPredictionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
};

TYPED_TEST_CASE(RegionPredictionLayerTest, TestDtypesAndDevices);

TYPED_TEST(RegionPredictionLayerTest, SameWithRegionOutput) {
  typedef typename TypeParam::Dtype Dtype;
  if (sizeof(Dtype) != 4) {
      return;
  }
  Blob<Dtype> blob_xy, blob_wh, blob_obj, blob_class, blob_imageinfo;
  int num = 1;
  int num_anchor = 3;
  int height = 13;
  int width = 13;
  int classes = 20;
  float nms = 0.45;
  float thresh = 0.005;
  std::vector<float> biases = {0.77871f, 1.14074f, 3.00525f, 4.31277f, 9.22725f, 9.61974f};

  // reshape
  blob_xy.Reshape(num, num_anchor * 2, height, width);
  blob_wh.ReshapeLike(blob_xy);
  blob_obj.Reshape(num, num_anchor, height, width);
  blob_class.Reshape(num, num_anchor * classes, height, width);
  blob_imageinfo.Reshape(1, 2, 1, 1);

  {
      Caffe::set_random_seed(777);
      // random populate the data
      FillerParameter filler_param;
      filler_param.set_min(0);
      filler_param.set_max(1);
      UniformFiller<Dtype> filler(filler_param);
      filler.Fill(&blob_xy);
      filler.Fill(&blob_wh);
      filler.Fill(&blob_obj);
      filler.Fill(&blob_class);
      blob_imageinfo.mutable_cpu_data()[0] = 416;
      blob_imageinfo.mutable_cpu_data()[1] = 416;
  }

  Blob<Dtype> region_top_bbox, region_top_prob;
  {
      // forward to the region outptu layer to get the result
      // create the input for the region layer
      Blob<Dtype> region_loss_input;
      region_loss_input.Reshape(num, num_anchor * (5 + classes), height, width);
      // copy the data into the region loss layout
      for (int b = 0; b < num; b++) {
          for (int n = 0; n < num_anchor; n++) {
              // x
              auto from = blob_xy.cpu_data() + blob_xy.offset(b, n, 0, 0);
              auto to = region_loss_input.mutable_cpu_data() + 
                  region_loss_input.offset(b, n * (5 + classes) + 0, 0, 0);
              caffe_copy(width * height, from, to);
                
              // y
              from = blob_xy.cpu_data() + blob_xy.offset(b, n + num_anchor, 0, 0);
              to = region_loss_input.mutable_cpu_data() + 
                  region_loss_input.offset(b, n * (5 + classes) + 1, 0, 0);
              caffe_copy(width * height, from, to);

              // w
              from = blob_wh.cpu_data() + blob_wh.offset(b, n, 0, 0);
              to = region_loss_input.mutable_cpu_data() +
                  region_loss_input.offset(b, n * (5 + classes) + 2, 0, 0);
              caffe_copy(width * height, from, to);

              // h
              from = blob_wh.cpu_data() + blob_wh.offset(b, n + num_anchor, 0, 0);
              to = region_loss_input.mutable_cpu_data() + 
                  region_loss_input.offset(b, n * (5 + classes) + 3, 0, 0);
              caffe_copy(width * height, from, to);

              // o
              from = blob_obj.cpu_data() + blob_obj.offset(b, n, 0, 0);
              to = region_loss_input.mutable_cpu_data() + 
                  region_loss_input.offset(b, n * (5 + classes) + 4, 0, 0);
              caffe_copy(width * height, from, to);

              // class
              from = blob_class.cpu_data() + blob_class.offset(b, n * classes, 0, 0);
              to = region_loss_input.mutable_cpu_data() + 
                  region_loss_input.offset(b, n * (5 + classes) + 5, 0, 0);
              caffe_copy(width * height * classes, from, to);
          }
      }
      LayerParameter region_output_param;
      RegionOutputParameter* region_param =
          region_output_param.mutable_region_output_param();
      for (int i = 0; i < biases.size(); i++) {
          region_param->add_biases(biases[i]);
      }
      region_param->set_classes(classes);
      region_param->set_nms(nms);
      region_param->set_thresh(thresh);
      ASSERT_EQ(region_param->biases_size() / 2, num_anchor);

      shared_ptr<RegionOutputLayer<Dtype>> region_output_layer(
              new RegionOutputLayer<Dtype>(region_output_param));
      std::vector<Blob<Dtype>* > region_output_bottom = {&region_loss_input, &blob_imageinfo};
      std::vector<Blob<Dtype>* > region_output_top = {&region_top_bbox, &region_top_prob};

      region_output_layer->LayerSetUp(region_output_bottom, region_output_top);
      region_output_layer->Reshape(region_output_bottom,    region_output_top);
      region_output_layer->Forward(region_output_bottom,    region_output_top);
    
      region_top_bbox.Reshape({num, num_anchor, height, width, 4});
      region_top_prob.Reshape({num, num_anchor, height, width, classes + 1});
  }

  {
    // sigmoid xy
    LayerParameter param_sig_xy;
    shared_ptr<SigmoidLayer<Dtype>> sig_layer_xy(
            new SigmoidLayer<Dtype>(param_sig_xy));
    std::vector<Blob<Dtype>*> sig_layer_xy_bottom;
    std::vector<Blob<Dtype>*> sig_layer_xy_top;
    sig_layer_xy_bottom.push_back(&blob_xy);
    sig_layer_xy_top.push_back(&blob_xy);
    sig_layer_xy->LayerSetUp(sig_layer_xy_bottom, sig_layer_xy_top);
    sig_layer_xy->Reshape(sig_layer_xy_bottom, sig_layer_xy_top);
    sig_layer_xy->Forward(sig_layer_xy_bottom, sig_layer_xy_top);
  }

  {
    // sigmoid o
    LayerParameter param_sig_obj;
    shared_ptr<SigmoidLayer<Dtype>> sig_layer_obj(
            new SigmoidLayer<Dtype>(param_sig_obj));
    std::vector<Blob<Dtype>*> sig_layer_o_bottom;
    std::vector<Blob<Dtype>*> sig_layer_o_top;
 
    sig_layer_o_bottom.push_back(&blob_obj);
    sig_layer_o_top.push_back(&blob_obj);
    sig_layer_obj->LayerSetUp(sig_layer_o_bottom, sig_layer_o_top);
    sig_layer_obj->Reshape(sig_layer_o_bottom, sig_layer_o_top);
    sig_layer_obj->Forward(sig_layer_o_bottom, sig_layer_o_top);
  }
  {
      // softmax class
      LayerParameter param_softmax_obj;
      shared_ptr<SoftmaxLayer<Dtype>> softmax_layer(new SoftmaxLayer<Dtype>(param_softmax_obj));
      blob_class.Reshape(num * num_anchor, classes, height, width);
      std::vector<Blob<Dtype>*> softmax_bottom = {&blob_class};
      std::vector<Blob<Dtype>*> softmax_top = {&blob_class};
      softmax_layer->LayerSetUp(softmax_bottom, softmax_top);
      softmax_layer->Reshape(softmax_bottom, softmax_top);
      softmax_layer->Forward(softmax_bottom, softmax_top);
  }
  Blob<Dtype> region_prediction_bbox, region_prediction_prob;
  {
      // forward to region predict layer
      LayerParameter param_predict;
      auto predict_param = param_predict.mutable_region_prediction_param();
      predict_param->set_nms(nms);
      for (int i = 0; i < biases.size(); i++) {
          predict_param->add_biases(biases[i]);
      }
      predict_param->set_thresh(thresh);

      shared_ptr<RegionPredictionLayer<Dtype>> region_prediction_layer(new RegionPredictionLayer<Dtype>(param_predict));
      std::vector<Blob<Dtype>*> region_prediction_bottom = {&blob_xy, &blob_wh, &blob_obj, &blob_class, &blob_imageinfo};
      std::vector<Blob<Dtype>*> region_prediction_top = {&region_prediction_bbox, &region_prediction_prob};
      region_prediction_layer->LayerSetUp(region_prediction_bottom, region_prediction_top);
      region_prediction_layer->Reshape(region_prediction_bottom, region_prediction_top);
      region_prediction_layer->Forward(region_prediction_bottom, region_prediction_top);
  }

  // assert
  for (int n = 0; n < num; n++) {
      for (int c = 0; c < num_anchor; c++) {
          for (int h = 0; h < height; h++) {
              for (int w = 0; w < width; w++) {
                  for (int d = 0; d < 4; d++) {
                      ASSERT_FLOAT_EQ(region_prediction_bbox.data_at({n, c, h, w, d}), 
                              region_top_bbox.data_at({n, c, h, w, d}))
                          << n << ", " << c << ", " << h << ", " << w << ", " << d;
                  }
                  for (int d = 0; d < classes + 1; d++) {
                      ASSERT_NEAR(region_prediction_prob.data_at({n, c, h, w, d}), 
                              region_top_prob.data_at({n, c, h, w, d}), 0.0001)
                          << n << ", " << c << ", " << h << ", " << w << ", " << d;
                  }
              }
          }
      }
  }
}
}
