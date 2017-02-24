// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Python code Written by Ross Girshick
// C++ implementation by Lei Zhang
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/layers/rpn_proposal_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void RPNProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, 
      const vector<Blob<Dtype>*>& top) {
    RPNProposalParameter rpn_proposal_param = this->layer_param_.rpn_proposal_param();
    CHECK_GT(rpn_proposal_param.feat_stride(), 0)
        << "feat_stride must be > 0";
    feat_stride_ = rpn_proposal_param.feat_stride();
    LOG(INFO) << "Feature stride: " << feat_stride_;
    anchors_.clear();
    anchors_.push_back(Rect(-84., -40., 99., 55.));
    anchors_.push_back(Rect(-176., -88., 191., 103.));
    anchors_.push_back(Rect(-360., -184., 375., 199.));
    anchors_.push_back(Rect(-56., -56., 71., 71.));
    anchors_.push_back(Rect(-120., -120., 135., 135.));
    anchors_.push_back(Rect(-248., -248., 263., 263.));
    anchors_.push_back(Rect(-36., -80., 51., 95.));
    anchors_.push_back(Rect(-80., -168., 95., 183.));
    anchors_.push_back(Rect(-168., -344., 183., 359.));

    //# rois blob : holds R regions of interest, each is a 5 - tuple
    //# (n, x1, y1, x2, y2) specifying an image batch index n and a
    //# rectangle(x1, y1, x2, y2)
    vector<int> shape(2);
    shape[0] = 1;
    shape[1] = 5;
    top[0]->Reshape(shape);

    //# scores blob : holds scores for R regions of interest
    if (top.size() > 1)
    {
        shape[1] = 1;
        top[1]->Reshape(shape);
    }
}

template <typename Dtype>
void RPNProposalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template<typename Dtype>
struct IdxCompare
{
    const std::vector<Dtype>& target;

    IdxCompare(const std::vector<Dtype>& target) : target(target) {}

    bool operator()(int a, int b) const { return target[a] > target[b]; }
};

template struct IdxCompare<float>;
template struct IdxCompare<double>;

template <typename Dtype>
void RPNProposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    CHECK_EQ(bottom[0]->shape(0), 1) << "Only single item batches are supported";

    int pre_nms_topN = 6000;
    int post_nms_topN = 300;
    float nms_thresh = 0.7;
    int min_size = 16;

    //# the first set of _num_anchors channels are bg probs
    //# the second set are the fg probs, which we want
    const Dtype *scores_data = bottom[0]->cpu_data();
    const Dtype *bbox_deltas_data = bottom[1]->cpu_data();
    const Dtype *im_info = bottom[2]->cpu_data();

    //# 1. Generate proposals from bbox deltas and shifted anchors
    int height = bottom[0]->shape(2);
    int width = bottom[0]->shape(3);

    //# Enumerate all shifts
    vector<Point> shifts;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            shifts.push_back(Point(x * feat_stride_, y * feat_stride_));

    //# Enumerate all shifted anchors :
    //#
    //# add A anchors(1, A, 4) to
    //# cell K shifts(K, 1, 4) to get
    //# shift anchors(K, A, 4)
    //# reshape to(K*A, 4) shifted anchors
    int A = anchors_.size();
    int K = shifts.size();
    vector<Rect> anchors;
    for (int i = 0; i < shifts.size(); i++)
        for (int j = 0; j < anchors_.size(); j++)
            anchors.push_back(anchors_[j].Shift(shifts[i].x, shifts[i].y));

//    # Transpose and reshape predicted bbox transformations to get them
//    # into the same order as the anchors :
//    #
//    # bbox deltas will be(1, 4 * A, H, W) format
//    # transpose to(1, H, W, 4 * A)
//    # reshape to(1 * H * W * A, 4) where rows are ordered by(h, w, a)
//    # in slowest to fastest order
    vector<Rect> bbox_deltas;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int a = 0; a < A; a++)
            {
                Dtype _x1 = bbox_deltas_data[(a * 4 + 0) * height * width + y * width + x];
                Dtype _y1 = bbox_deltas_data[(a * 4 + 1) * height * width + y * width + x];
                Dtype _x2 = bbox_deltas_data[(a * 4 + 2) * height * width + y * width + x];
                Dtype _y2 = bbox_deltas_data[(a * 4 + 3) * height * width + y * width + x];
                bbox_deltas.push_back(Rect(_x1, _y1, _x2, _y2));
            }

    //# Same story for the scores :
    //#
    //# scores are(1, A, H, W) format
    //# transpose to(1, H, W, A)
    //# reshape to(1 * H * W * A, 1) where rows are ordered by(h, w, a)
    vector<Dtype> scores;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int a = 0; a < A; a++)
            {
                //# the first set of _num_anchors channels are bg probs
                //# the second set are the fg probs, which we want
                Dtype s = scores_data[(A + a) * height * width + y * width + x];
                scores.push_back(s);
            }

    //# Convert anchors into proposals via bbox transformations
    vector<Rect> proposals = bbox_transform_inv(anchors, bbox_deltas);

    //# 2. clip predicted boxes to image
    Dtype im_height = im_info[0];
    Dtype im_weight = im_info[1];
    for (int i = 0; i < proposals.size(); i++)
    {
        Rect &box = proposals[i];
        //# x1 >= 0
        box.x1 = max(min(box.x1, im_weight - 1), (Dtype)0);
        //# y1 >= 0
        box.y1 = max(min(box.y1, im_height - 1), (Dtype)0);
        //# x2 < im_shape[1]
        box.x2 = max(min(box.x2, im_weight - 1), (Dtype)0);
        //# y2 < im_shape[0]
        box.y2 = max(min(box.y2, im_height - 1), (Dtype)0);
    }

    //# 3. remove predicted boxes with either height or width < threshold
    //# (NOTE: convert min_size to input image scale stored in im_info[2])
    vector<Rect> _proposals;
    vector<Dtype> _scores;
    for (int i = 0; i < proposals.size(); i++)
    {
        Rect &box = proposals[i];
        if (box.Width() >= min_size * im_info[2] && box.Height() >= min_size * im_info[2])
        {
            _proposals.push_back(box);
            _scores.push_back(scores[i]);
        }
    }
    proposals = _proposals;
    scores = _scores;

    //# 4. sort all(proposal, score) pairs by score from highest to lowest
    //# 5. take top pre_nms_topN(e.g. 6000)
    vector<int> order;
    for (int i = 0; i < proposals.size(); ++i)
        order.push_back(i);
    std::sort(order.begin(), order.end(), IdxCompare<Dtype>(scores));
    if (pre_nms_topN > 0)
        order.resize(pre_nms_topN);
    _proposals.resize(order.size());
    _scores.resize(order.size());
    for (int i = 0; i < order.size(); i++)
    {
        _proposals[i] = proposals[order[i]];
        _scores[i] = scores[order[i]];
    }
    proposals = _proposals;
    scores = _scores;

    //# 6. apply nms(e.g.threshold = 0.7)
    //# 7. take after_nms_topN(e.g. 300)
    //# 8. return the top proposals(->RoIs top)
    vector<bool> suppressed(proposals.size());
    std::fill(suppressed.begin(), suppressed.end(), false);
    vector<int> keep;
    for (int i = 0; i < proposals.size(); i++)
    {
        if (suppressed[i])
            continue;
        keep.push_back(i);
        Rect &_i = proposals[i];
        Dtype iarea = _i.Area();
        for (int j = i + 1; j < proposals.size(); j++)
        {
            if (suppressed[j])
                continue;
            Rect &_j = proposals[j];
            Dtype xx1 = max(_i.x1, _j.x1);
            Dtype yy1 = max(_i.y1, _j.y1);
            Dtype xx2 = min(_i.x2, _j.x2);
            Dtype yy2 = min(_i.y2, _j.y2);
            Dtype w = max((Dtype)0.0, xx2 - xx1 + 1);
            Dtype h = max((Dtype)0.0, yy2 - yy1 + 1);
            Dtype inter = w * h;
            Dtype ovr = inter / (iarea + _j.Area() - inter);
            if (ovr >= nms_thresh)
                suppressed[j] = true;
        }
    }
    keep.resize(min(post_nms_topN, (int)keep.size()));

    //# Output rois blob
    //# Our RPN implementation only supports a single input image, so all
    //# batch inds are 0
    vector<int> shape(2);
    shape[0] = keep.size();
    shape[1] = 5;
    top[0]->Reshape(shape);
    Dtype *rois_data = top[0]->mutable_cpu_data();
    memset(rois_data, 0, sizeof(Dtype) * top[0]->count());  // clear batch inds
    for (int i = 0; i < keep.size(); i++)
        memcpy(rois_data + i * 5 + 1, &proposals[keep[i]], sizeof(Dtype) * 4);

    if (top.size() > 1)
    {
        shape[1] = 1;
        top[1]->Reshape(shape);
        Dtype *scores_data = top[1]->mutable_cpu_data();
        for (int i = 0; i < keep.size(); i++)
            scores_data[i] = scores[keep[i]];
    }
}

template <typename Dtype>
void RPNProposalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_CLASS(RPNProposalLayer);
REGISTER_LAYER_CLASS(RPNProposal);

}  // namespace caffe
