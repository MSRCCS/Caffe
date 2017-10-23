#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/region_target_layer.hpp"
#include "caffe/layers/region_loss_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/layers/softmax_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class RegionTargetLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  RegionTargetLayerTest(): blob_bottom_feature_(new Blob<Dtype>()),
    blob_gt_(new Blob<Dtype>()),
    blob_top_loss_(new Blob<Dtype>()) {
  }
  void SigmoidForward(Blob<Dtype> &blob) {
    LayerParameter param_sig_xy;
    shared_ptr<SigmoidLayer<Dtype>> sig_layer_xy(
            new SigmoidLayer<Dtype>(param_sig_xy));
    std::vector<Blob<Dtype>*> sig_layer_xy_bottom;
    std::vector<Blob<Dtype>*> sig_layer_xy_top;
    sig_layer_xy_bottom.push_back(&blob);
    sig_layer_xy_top.push_back(&blob);
    sig_layer_xy->LayerSetUp(sig_layer_xy_bottom, sig_layer_xy_top);
    sig_layer_xy->Reshape(sig_layer_xy_bottom, sig_layer_xy_top);
    sig_layer_xy->Forward(sig_layer_xy_bottom, sig_layer_xy_top);
  }
  virtual ~RegionTargetLayerTest() {
    delete blob_bottom_feature_;
    delete blob_gt_;
    delete blob_top_loss_;
  }
  Blob<Dtype>* const blob_bottom_feature_;
  Blob<Dtype>* const blob_gt_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(RegionTargetLayerTest, TestDtypesAndDevices);

TYPED_TEST(RegionTargetLayerTest, IdenticalWithRegionLoss) {
  typedef typename TypeParam::Dtype Dtype;
  if (sizeof(Dtype) != 4) {
      return;
  }
  int num = 6;
  int width = 13;
  int height = 13;
  int num_gt = 30;
  float gt_label = 2;
  int classes = 20;
  vector<float> biases = {0.77871f, 1.14074f, 3.00525f, 4.31277f,
                          9.22725f, 9.61974f};

  int num_anchor = biases.size() / 2;

  // define the input
  Blob<Dtype> blob_xy, blob_wh, blob_obj, blob_truth, blob_class;
  blob_xy.Reshape(num, 2 * num_anchor, height, width);
  blob_wh.Reshape(num, 2 * num_anchor, height, width);
  blob_obj.Reshape(num, num_anchor, height, width);
  blob_class.Reshape(num, num_anchor * classes, height, width);
  blob_truth.Reshape(num, num_gt * 5, 1, 1);

  // fill the input
  Caffe::set_random_seed(777);
  FillerParameter filler_param;
  filler_param.set_min(0);
  filler_param.set_max(1);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(&blob_xy);
  //caffe_set(blob_xy.count(), Dtype(0), blob_xy.mutable_cpu_data());
  filler.Fill(&blob_wh);
  filler.Fill(&blob_obj);
  filler.Fill(&blob_truth);
  for (int n = 0; n < num; n++) {
      for (int j = 0; j < num_gt; j++) {
            blob_truth.mutable_cpu_data()[n * 150 + j * 5 + 4] = gt_label;
      }
  }

  // define the output
  Blob<Dtype> target_xy, target_wh, target_obj_obj, target_obj_noobj, target_class;
  Blob<Dtype> target_xywh_weight;

  // forward and backward it to region loss layer
  Blob<Dtype> region_loss_layer_bottom_diff;
  {
      LayerParameter region_loss_param;
      RegionLossParameter* region_param =
          region_loss_param.mutable_region_loss_param();
      region_param->set_classes(classes);
      region_param->set_coords(4);
      region_param->set_bias_match(true);
      region_param->set_softmax(true);
      region_param->set_rescore(true);
      region_param->set_object_scale(5.0);
      for (int i = 0; i < biases.size(); i++) {
          region_param->add_biases(biases[i]);
      }
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

      // set up the layer
      shared_ptr<RegionLossLayer<Dtype>> region_loss_layer(
              new RegionLossLayer<Dtype>(region_loss_param));
      std::vector<Blob<Dtype>* > region_loss_bottom = {&region_loss_input, &blob_truth};
      Blob<Dtype> region_top_blob;
      region_top_blob.Reshape(1, 1, 1, 1);
      *region_top_blob.mutable_cpu_diff() = 1.0;
      std::vector<Blob<Dtype>* > region_loss_top = {&region_top_blob};
      region_loss_layer->LayerSetUp(region_loss_bottom, region_loss_top);
      region_loss_layer->Reshape(region_loss_bottom, region_loss_top);
      region_loss_layer->Forward(region_loss_bottom, region_loss_top);
      region_loss_layer->Backward(region_loss_top, {true, false}, region_loss_bottom);

      region_loss_layer_bottom_diff.ReshapeLike(region_loss_input);
      region_loss_layer_bottom_diff.CopyFrom(region_loss_input, true);
  }

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

  // sigmoid o
  LayerParameter param_sig_obj;
  shared_ptr<SigmoidLayer<Dtype>> sig_layer_obj(
          new SigmoidLayer<Dtype>(param_sig_obj));
  std::vector<Blob<Dtype>*> sig_layer_o_bottom;
  std::vector<Blob<Dtype>*> sig_layer_o_top;
  {
    sig_layer_o_bottom.push_back(&blob_obj);
    sig_layer_o_top.push_back(&blob_obj);
    sig_layer_obj->LayerSetUp(sig_layer_o_bottom, sig_layer_o_top);
    sig_layer_obj->Reshape(sig_layer_o_bottom, sig_layer_o_top);
    sig_layer_obj->Forward(sig_layer_o_bottom, sig_layer_o_top);
  }

  {
    // region target
    LayerParameter region_target_layer_param;
    RegionTargetParameter* region_param =
        region_target_layer_param.mutable_region_target_param();
    for (int i = 0; i < biases.size(); i++) {
        region_param->add_biases(biases[i]);
    }
    shared_ptr<RegionTargetLayer<Dtype>> target_layer(
            new RegionTargetLayer<Dtype>(region_target_layer_param));
    std::vector<Blob<Dtype>* > target_layer_bottom = {
        &blob_xy, &blob_wh, &blob_obj, &blob_truth };
    std::vector<Blob<Dtype>* > target_layer_top = {&target_xy, &target_wh, 
          &target_xywh_weight,
        &target_obj_obj, &target_obj_noobj, &target_class};
    target_layer->LayerSetUp(target_layer_bottom, target_layer_top);
    target_layer->Reshape(target_layer_bottom, target_layer_top);
    target_layer->Forward(target_layer_bottom, target_layer_top);
  }

  {
    // loss xy
    LayerParameter euc_loss_xy_layer_param;
    shared_ptr<EuclideanLossLayer<Dtype>> eu_loss_xy_layer(new EuclideanLossLayer<Dtype>(euc_loss_xy_layer_param));
    Blob<Dtype> eu_loss_xy_blob_top;
    eu_loss_xy_blob_top.Reshape(1, 1, 1, 1);
    eu_loss_xy_blob_top.mutable_cpu_diff()[0] = 1;
    std::vector<Blob<Dtype>*> eu_loss_xy_layer_bottom = {&blob_xy, &target_xy, &target_xywh_weight};
    std::vector<Blob<Dtype>*> eu_loss_xy_layer_top = {&eu_loss_xy_blob_top};
    eu_loss_xy_layer->LayerSetUp(eu_loss_xy_layer_bottom, eu_loss_xy_layer_top);
    eu_loss_xy_layer->Reshape(eu_loss_xy_layer_bottom, eu_loss_xy_layer_top);
    eu_loss_xy_layer->Forward(eu_loss_xy_layer_bottom, eu_loss_xy_layer_top);
    eu_loss_xy_layer->Backward(eu_loss_xy_layer_top, {true, false, false}, eu_loss_xy_layer_bottom);

    // back xy from sigmoid
    sig_layer_xy->Backward(sig_layer_xy_top, {true}, sig_layer_xy_bottom);
  }

  {
    // loss wh
    LayerParameter euc_loss_wh_layer_param;
    shared_ptr<EuclideanLossLayer<Dtype>> eu_loss_wh_layer(new EuclideanLossLayer<Dtype>(euc_loss_wh_layer_param));
    Blob<Dtype> eu_loss_wh_blob_top;
    eu_loss_wh_blob_top.Reshape(1, 1, 1, 1);
    eu_loss_wh_blob_top.mutable_cpu_diff()[0] = 1;
    std::vector<Blob<Dtype>*> eu_loss_wh_layer_bottom = {&blob_wh, &target_wh, &target_xywh_weight};
    std::vector<Blob<Dtype>*> eu_loss_wh_layer_top = {&eu_loss_wh_blob_top};
    eu_loss_wh_layer->LayerSetUp(eu_loss_wh_layer_bottom, eu_loss_wh_layer_top);
    eu_loss_wh_layer->Reshape(eu_loss_wh_layer_bottom, eu_loss_wh_layer_top);
    eu_loss_wh_layer->Forward(eu_loss_wh_layer_bottom, eu_loss_wh_layer_top);
    eu_loss_wh_layer->Backward(eu_loss_wh_layer_top, {true, false}, eu_loss_wh_layer_bottom);
  }

  // check xy and wh
  Dtype s_xy = 0;
  Dtype s_wh = 0;
  Dtype s_obj = 0;
  Dtype s_class = 0;
  for (int b = 0; b < num; b++) {
      for (int n = 0; n < num_anchor; n++) {
          auto p = region_loss_layer_bottom_diff.cpu_diff() + region_loss_layer_bottom_diff.offset(b, (5 + classes) * n, 0, 0);
          s_xy += caffe_cpu_asum(2 * height * width, p);
          p = region_loss_layer_bottom_diff.cpu_diff() + region_loss_layer_bottom_diff.offset(b, (5 + classes) * n + 2, 0, 0);
          s_wh += caffe_cpu_asum(2 * height * width, p);
          p = region_loss_layer_bottom_diff.cpu_diff() + region_loss_layer_bottom_diff.offset(b, (5 + classes) * n + 4, 0, 0);
          s_obj += caffe_cpu_asum(height * width, p);
          p = region_loss_layer_bottom_diff.cpu_diff() + region_loss_layer_bottom_diff.offset(b, (5 + classes) * n + 5, 0, 0);
          s_class += caffe_cpu_asum(height * width * classes, p);
      }
  }
  EXPECT_FLOAT_EQ(blob_xy.asum_diff(), s_xy);
  EXPECT_FLOAT_EQ(blob_wh.asum_diff(), s_wh);
  Dtype largest_diff = 0;
  Dtype largest_b, largest_n, largest_j, largest_i;
  for (int b = 0; b < num; b++) {
      for (int n = 0; n < num_anchor; n++) {
          for (int j = 0; j < height; j++) {
              for (int i = 0; i < width; i++) {
                auto c = blob_xy.diff_at(b, n, j, i);
                auto t = region_loss_layer_bottom_diff.diff_at(b, n * (classes + 5), j, i);
                auto diff = c - t;
                diff = diff > 0 ? diff : -diff;
                if (diff > largest_diff) {
                    largest_diff = diff;
                    largest_b = b;
                    largest_n = n;
                    largest_j = j;
                    largest_i = i;
                }
              }
          }
      }
  }
  EXPECT_NEAR(largest_diff, 0, 0.0001) << largest_b << "," 
      << largest_n << "," << largest_j << "," << largest_i;

  Blob<Dtype> blob_obj1;
  blob_obj1.ReshapeLike(blob_obj);
  blob_obj1.CopyFrom(blob_obj);
  {
    // loss obj-obj
    LayerParameter euc_loss_o_layer_param1;
    shared_ptr<EuclideanLossLayer<Dtype>> eu_loss_o_layer1(new EuclideanLossLayer<Dtype>(euc_loss_o_layer_param1));
    Blob<Dtype> eu_loss_o_blob_top1;
    eu_loss_o_blob_top1.Reshape(1, 1, 1, 1);
    eu_loss_o_blob_top1.mutable_cpu_diff()[0] = 5;
    std::vector<Blob<Dtype>*> eu_loss_o_layer_bottom1 = {&blob_obj1, &target_obj_obj};
    std::vector<Blob<Dtype>*> eu_loss_o_layer_top1 = {&eu_loss_o_blob_top1};
    eu_loss_o_layer1->LayerSetUp(eu_loss_o_layer_bottom1, eu_loss_o_layer_top1);
    eu_loss_o_layer1->Reshape(eu_loss_o_layer_bottom1, eu_loss_o_layer_top1);
    eu_loss_o_layer1->Forward(eu_loss_o_layer_bottom1, eu_loss_o_layer_top1);
    eu_loss_o_layer1->Backward(eu_loss_o_layer_top1, {true, false}, eu_loss_o_layer_bottom1);
  }

  Blob<Dtype> blob_obj2;
  blob_obj2.ReshapeLike(blob_obj);
  blob_obj2.CopyFrom(blob_obj);
  {
    // loss obj-noobj
    LayerParameter euc_loss_o_layer_param2;
    shared_ptr<EuclideanLossLayer<Dtype>> eu_loss_o_layer2(new EuclideanLossLayer<Dtype>(euc_loss_o_layer_param2));
    Blob<Dtype> eu_loss_o_blob_top2;
    eu_loss_o_blob_top2.Reshape(1, 1, 1, 1);
    eu_loss_o_blob_top2.mutable_cpu_diff()[0] = 1;
    std::vector<Blob<Dtype>*> eu_loss_o_layer_bottom2 = {&blob_obj2, &target_obj_noobj};
    std::vector<Blob<Dtype>*> eu_loss_o_layer_top2 = {&eu_loss_o_blob_top2};
    eu_loss_o_layer2->LayerSetUp(eu_loss_o_layer_bottom2, eu_loss_o_layer_top2);
    eu_loss_o_layer2->Reshape(eu_loss_o_layer_bottom2, eu_loss_o_layer_top2);
    eu_loss_o_layer2->Forward(eu_loss_o_layer_bottom2, eu_loss_o_layer_top2);
    eu_loss_o_layer2->Backward(eu_loss_o_layer_top2, {true, false}, eu_loss_o_layer_bottom2);
  }

  // obj: merge the two
  caffe_add(blob_obj.count(), blob_obj1.cpu_diff(), blob_obj2.cpu_diff(), blob_obj.mutable_cpu_diff());
  sig_layer_obj->Backward(sig_layer_o_top, {true}, sig_layer_o_bottom);
  EXPECT_FLOAT_EQ(blob_obj.asum_diff(), s_obj);

  // class
  blob_class.Reshape(num * num_anchor, classes, height, width);
  target_class.Reshape(num * num_anchor, 1, height, width);
  {
      LayerParameter softmax_loss_param;
      auto loss_param = softmax_loss_param.mutable_loss_param();
      loss_param->set_ignore_label(-1);
      loss_param->set_normalization(LossParameter_NormalizationMode_BATCH_SIZE);
      shared_ptr<SoftmaxWithLossLayer<Dtype>> softmax_loss_layer(new SoftmaxWithLossLayer<Dtype>(softmax_loss_param));
      Blob<Dtype> softmax_loss_blob_top;
      softmax_loss_blob_top.Reshape(1, 1, 1, 1);
      softmax_loss_blob_top.mutable_cpu_diff()[0] = num_anchor;
      std::vector<Blob<Dtype>*> softmax_loss_bottom = {&blob_class, &target_class};
      std::vector<Blob<Dtype>*> softmax_loss_top = {&softmax_loss_blob_top};
      softmax_loss_layer->LayerSetUp(softmax_loss_bottom, softmax_loss_top);
      softmax_loss_layer->Reshape(softmax_loss_bottom, softmax_loss_top);
      softmax_loss_layer->Forward(softmax_loss_bottom, softmax_loss_top);
      softmax_loss_layer->Backward(softmax_loss_top, {true, false}, softmax_loss_bottom);
  }

  EXPECT_NEAR(s_class, blob_class.asum_diff(), s_class * 0.0001);
}

TYPED_TEST(RegionTargetLayerTest, OneGT) {
  typedef typename TypeParam::Dtype Dtype;
  int num = 6;
  int width = 13;
  int height = 13;
  int num_gt = 15;
  float gt_x = 0.5;
  float gt_y = 0.1;
  int gt_aligned_biases = 1;
  float gt_label = 2;
  vector<float> biases = {0.77871f, 1.14074f, 3.00525f, 4.31277f,
                          9.22725f, 9.61974f};
  float gt_w = biases[gt_aligned_biases * 2] / width;
  float gt_h = biases[gt_aligned_biases * 2 + 1] / height;

  int num_anchor = biases.size() / 2;

  Blob<Dtype> blob_xy, blob_wh, blob_obj, blob_truth;
  blob_xy.Reshape(num, 2 * num_anchor, height, width);
  blob_wh.Reshape(num, 2 * num_anchor, height, width);
  blob_obj.Reshape(num, num_anchor, height, width);
  blob_truth.Reshape(num, num_gt * 5, 1, 1);

  Caffe::set_random_seed(777);
  FillerParameter filler_param;
  filler_param.set_min(0);
  filler_param.set_max(1);
  UniformFiller<Dtype> filler(filler_param);
  filler.Fill(&blob_xy);
  filler.Fill(&blob_wh);
  filler.Fill(&blob_obj);
  caffe_set(blob_truth.count(), Dtype(0), blob_truth.mutable_cpu_data());
  blob_truth.mutable_cpu_data()[0] = gt_x;
  blob_truth.mutable_cpu_data()[1] = gt_y;
  blob_truth.mutable_cpu_data()[2] = gt_w;
  blob_truth.mutable_cpu_data()[3] = gt_h;
  blob_truth.mutable_cpu_data()[4] = gt_label;

  Blob<Dtype> target_xy, target_wh, target_obj_obj, target_obj_noobj, target_class;
    Blob<Dtype> target_xywh_weight;

  // sigmod xy
  this->SigmoidForward(blob_xy);
  this->SigmoidForward(blob_obj);

  // setup the layer param
  LayerParameter layer_param;
  RegionTargetParameter* region_param =
      layer_param.mutable_region_target_param();
  region_param->set_thresh(0.6);
  region_param->set_rescore(true);
  region_param->set_anchor_aligned_images(0);
  for (int i = 0; i < biases.size(); i++) {
      region_param->add_biases(biases[i]);
  }

  shared_ptr<RegionTargetLayer<Dtype>> target_layer(
          new RegionTargetLayer<Dtype>(layer_param));
  std::vector<Blob<Dtype>* > target_layer_bottom = {
      &blob_xy, &blob_wh, &blob_obj, &blob_truth };
  std::vector<Blob<Dtype>* > target_layer_top = {&target_xy, &target_wh, 
        &target_xywh_weight,
      &target_obj_obj, &target_obj_noobj, &target_class};
  target_layer->LayerSetUp(target_layer_bottom, target_layer_top);
  target_layer->Reshape(target_layer_bottom, target_layer_top);
  target_layer->Forward(target_layer_bottom, target_layer_top);

  EXPECT_FLOAT_EQ(target_xy.data_at(0, gt_aligned_biases, gt_y * height, gt_x * width),
          gt_x * width - (int)(gt_x * width));
  EXPECT_FLOAT_EQ(target_xy.data_at(0, num_anchor + gt_aligned_biases, gt_y * height, gt_x * width),
          gt_y * height - (int)(gt_y * height));
  *(target_xy.mutable_cpu_data() + target_xy.offset(0, gt_aligned_biases, gt_y * height, gt_x * width)) = 
      blob_xy.data_at(0, gt_aligned_biases, gt_y * height, gt_x * width);
  *(target_xy.mutable_cpu_data() + target_xy.offset(0, num_anchor + gt_aligned_biases, gt_y * height, gt_x * width)) = 
      blob_xy.data_at(0, gt_aligned_biases + num_anchor, gt_y * height, gt_x * width);
  EXPECT_FLOAT_EQ(target_xy.asum_data(), blob_xy.asum_data());

  EXPECT_NEAR(target_wh.data_at(0, gt_aligned_biases, gt_y * height, gt_x * width), 0, 0.00001);
  EXPECT_NEAR(target_wh.data_at(0, gt_aligned_biases + num_anchor, gt_y * height, gt_x * width), 0, 0.00001);
  *(target_wh.mutable_cpu_data() + target_wh.offset(0, gt_aligned_biases, gt_y * height, gt_x * width)) = 
      blob_wh.data_at(0, gt_aligned_biases, gt_y * height, gt_x * width);
  *(target_wh.mutable_cpu_data() + target_wh.offset(0, gt_aligned_biases + num_anchor, gt_y * height, gt_x * width)) = 
      blob_wh.data_at(0, gt_aligned_biases + num_anchor, gt_y * height, gt_x * width);
  EXPECT_FLOAT_EQ(target_wh.asum_data(), blob_wh.asum_data());

  EXPECT_FLOAT_EQ(target_class.data_at(0, gt_aligned_biases, gt_y * height, gt_x * width), gt_label);
  *(target_class.mutable_cpu_data() + target_class.offset(0, gt_aligned_biases, gt_y * height, gt_x * width)) = -1;
  EXPECT_FLOAT_EQ(target_class.asum_data(), 1 * target_class.count());

  EXPECT_GT(target_obj_obj.data_at(0, gt_aligned_biases, gt_y * height, gt_x * width), 0);
  *(target_obj_obj.mutable_cpu_data() + target_obj_obj.offset(0, gt_aligned_biases, gt_y * height, gt_x * width)) = 
      blob_obj.data_at(0, gt_aligned_biases, gt_y * height, gt_x * width);
  EXPECT_FLOAT_EQ(target_obj_obj.asum_data(), blob_obj.asum_data());

  //EXPECT_FLOAT_EQ(target_obj_noobj.asum_data(), 0);
}

TYPED_TEST(RegionTargetLayerTest, NoGroundTruthNoAlign) {
    typedef typename TypeParam::Dtype Dtype;
    int num = 6;
    int width = 13;
    int height = 13;
    int num_gt = 15;
    vector<float> biases = {0.77871f, 1.14074f, 3.00525f, 4.31277f,
        9.22725f, 9.61974f};

    int num_anchor = biases.size() / 2;

    Blob<Dtype> blob_xy, blob_wh, blob_obj, blob_truth;
    blob_xy.Reshape(num, 2 * num_anchor, height, width);
    blob_wh.Reshape(num, 2 * num_anchor, height, width);
    blob_obj.Reshape(num, num_anchor, height, width);
    blob_truth.Reshape(num, num_gt * 5, 1, 1);

    Caffe::set_random_seed(777);
    FillerParameter filler_param;
    filler_param.set_min(0);
    filler_param.set_max(1);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(&blob_xy);
    filler.Fill(&blob_wh);
    filler.Fill(&blob_obj);
    caffe_set(blob_truth.count(), Dtype(0), blob_truth.mutable_cpu_data());

    Blob<Dtype> target_xy, target_wh, target_obj_obj, target_obj_noobj, target_class;
    Blob<Dtype> target_xywh_weight;

    // sigmod xy
    this->SigmoidForward(blob_xy);
    this->SigmoidForward(blob_obj);

    // setup the layer param
    LayerParameter layer_param;
    RegionTargetParameter* region_param =
        layer_param.mutable_region_target_param();
    region_param->set_thresh(0.6);
    region_param->set_rescore(true);
    region_param->set_anchor_aligned_images(0);
    for (int i = 0; i < biases.size(); i++) {
        region_param->add_biases(biases[i]);
    }

    shared_ptr<RegionTargetLayer<Dtype>> target_layer(
            new RegionTargetLayer<Dtype>(layer_param));
    std::vector<Blob<Dtype>* > target_layer_bottom = {
        &blob_xy, &blob_wh, &blob_obj, &blob_truth };
    std::vector<Blob<Dtype>* > target_layer_top = {&target_xy, &target_wh, 
        &target_xywh_weight,
        &target_obj_obj, &target_obj_noobj, &target_class};
    target_layer->LayerSetUp(target_layer_bottom, target_layer_top);
    target_layer->Reshape(target_layer_bottom, target_layer_top);
    target_layer->Forward(target_layer_bottom, target_layer_top);

    EXPECT_FLOAT_EQ(target_obj_noobj.asum_data(), 0);
    EXPECT_FLOAT_EQ(target_obj_obj.asum_data(), blob_obj.asum_data());
    EXPECT_FLOAT_EQ(target_xy.asum_data(), blob_xy.asum_data());
    EXPECT_FLOAT_EQ(target_wh.asum_data(), blob_wh.asum_data());
    EXPECT_FLOAT_EQ(target_class.asum_data(), 1 * target_class.count());
}
TYPED_TEST(RegionTargetLayerTest, NoGroundTruth) {
    typedef typename TypeParam::Dtype Dtype;
    int num = 6;
    int width = 13;
    int height = 13;
    int num_gt = 15;
    vector<float> biases = {0.77871f, 1.14074f, 3.00525f, 4.31277f,
        9.22725f, 9.61974f};

    int num_anchor = biases.size() / 2;

    Blob<Dtype> blob_xy, blob_wh, blob_obj, blob_truth;
    blob_xy.Reshape(num, 2 * num_anchor, height, width);
    blob_wh.Reshape(num, 2 * num_anchor, height, width);
    blob_obj.Reshape(num, num_anchor, height, width);
    blob_truth.Reshape(num, num_gt * 5, 1, 1);

    Caffe::set_random_seed(777);
    FillerParameter filler_param;
    filler_param.set_min(0);
    filler_param.set_max(1);
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(&blob_xy);
    filler.Fill(&blob_wh);
    filler.Fill(&blob_obj);
    caffe_set(blob_truth.count(), Dtype(0), blob_truth.mutable_cpu_data());

    Blob<Dtype> target_xy, target_wh, target_obj_obj, target_obj_noobj, target_class;
    Blob<Dtype> target_xywh_weight;

    // sigmod xy
    this->SigmoidForward(blob_xy);
    this->SigmoidForward(blob_obj);

    // setup the layer param
    LayerParameter layer_param;
    RegionTargetParameter* region_param =
        layer_param.mutable_region_target_param();
    region_param->set_thresh(0.6);
    region_param->set_rescore(true);
    for (int i = 0; i < biases.size(); i++) {
        region_param->add_biases(biases[i]);
    }

    shared_ptr<RegionTargetLayer<Dtype>> target_layer(
            new RegionTargetLayer<Dtype>(layer_param));
    std::vector<Blob<Dtype>* > target_layer_bottom = {
        &blob_xy, &blob_wh, &blob_obj, &blob_truth };
    std::vector<Blob<Dtype>* > target_layer_top = {&target_xy, &target_wh, 
        &target_xywh_weight,
        &target_obj_obj, &target_obj_noobj, &target_class};
    target_layer->LayerSetUp(target_layer_bottom, target_layer_top);
    target_layer->Reshape(target_layer_bottom, target_layer_top);
    target_layer->Forward(target_layer_bottom, target_layer_top);

    EXPECT_FLOAT_EQ(target_obj_noobj.asum_data(), 0);
    EXPECT_FLOAT_EQ(target_obj_obj.asum_data(), blob_obj.asum_data());
    EXPECT_FLOAT_EQ(target_xy.asum_data(), 0.5 * target_xy.count());
    EXPECT_FLOAT_EQ(target_wh.asum_data(), 0);
    EXPECT_FLOAT_EQ(target_class.asum_data(), 1 * target_class.count());
}
}  // namespace caffe
