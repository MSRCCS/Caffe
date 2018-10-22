#ifndef CAFFE_REGION_COMMON_HPP_
#define CAFFE_REGION_COMMON_HPP_

namespace caffe {

template <typename Dtype>
struct TBox {
    Dtype x, y, w, h;
};

template <typename Dtype> 
__host__ __device__ Dtype TOverlap(Dtype x1, Dtype w1, Dtype x2, Dtype w2)
{
    auto l1 = x1 - w1 / 2;
    auto l2 = x2 - w2 / 2;
    auto left = l1 > l2 ? l1 : l2;
    auto r1 = x1 + w1 / 2;
    auto r2 = x2 + w2 / 2;
    auto right = r1 < r2 ? r1 : r2;
    return right - left;
}

template <typename Dtype>
__host__ __device__ Dtype TBoxIntersection(Dtype ax, Dtype ay, Dtype aw, Dtype ah, 
        Dtype bx, Dtype by, Dtype bw, Dtype bh) {
    auto w = TOverlap(ax, aw, bx, bw);
    auto h = TOverlap(ay, ah, by, bh);
    if (w < 0 || h < 0) {
        return 0;
    }
    else {
        return w * h;
    }
}

template <typename Dtype>
__host__ __device__ Dtype TBoxUnion(Dtype aw, Dtype ah,
                                    Dtype bw, Dtype bh,
                                    Dtype i) {
    auto u = aw * ah + bw * bh - i;
    return u;
}

template <typename Dtype>
__host__ __device__ Dtype TBoxIou(Dtype ax, Dtype ay, Dtype aw, Dtype ah, 
        Dtype bx, Dtype by, Dtype bw, Dtype bh) {
    auto i = TBoxIntersection(ax, ay, aw, ah, bx, by, bw, bh);
    auto u = aw * ah + bw * bh - i;
    return i / u;
}

}
#endif  // CAFFE_EUCLIDEAN_LOSS_LAYER_HPP_
