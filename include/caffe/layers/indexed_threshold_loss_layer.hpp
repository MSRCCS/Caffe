#ifndef CAFFE_INDEXEDTHRESHOLD_LOSS_LAYER_HPP_
#define CAFFE_INDEXEDTHRESHOLD_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
* @brief Computes the Euclidean (L2) loss with a positive threshold @f$
*          E = \frac{1}{2N} \sum\limits_{n=1}^N \left| \left| \hat{y}_n - y_n
*        \right| \right|_2^2 @f$ if @f$ \hat{y}_n < threshold  @f$
*        @f$ E = 0 @f otherwise.$
*
* @param bottom input Blob vector (length 2)
*   -# @f$ (N \times C \times H \times W) @f$
*      the predictions @f$ \hat{y} \in [-\infty, +\infty]@f$
*   -# @f$ (N) @f$
*      the indices @f$ n \in [0, C \times H \times W - 1]@f$ of positive targets
* @param top output Blob vector (length 1)
*   -# @f$ (1 \times 1 \times 1 \times 1) @f$
*      the computed Euclidean loss: @f$ E =
*          \frac{1}{2n} \sum\limits_{n=1}^N \left| \left| \hat{y}_n - y_n
*        \right| \right|_2^2 @f$ for @f$ \hat{y}_n < threshold  @f$
*/
template <typename Dtype>
class IndexedThresholdLossLayer : public LossLayer<Dtype> {
public:
    explicit IndexedThresholdLossLayer(const LayerParameter& param)
        : LossLayer<Dtype>(param), diff_() {
    }
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const {
        return "IndexedThresholdLoss";
    }

protected:
    /// @copydoc IndexedThresholdLossLayer
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);

    /**
    * @brief Computes the thresholded Euclidean error gradient w.r.t. the inputs.
    *
    * @param param provides IndexedThresholdLossParameter indexedthreshold_loss_param, with options:
    *  - threshold (optional)
    *    Specify the threshold only below which loss is non-zero for given indices.
    *  - null_scale (optional)
    *    Specify the scale to apply for loss of the zero/null targets
    *  - positive_scale (optional)
    *    Specify the scale to apply for loss of the positive targets (given by the indices)
    *
    * @param top output Blob vector (length 1), providing the error gradient with
    *      respect to the outputs
    *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
    *      This Blob's diff will simply contain the loss_weight* @f$ \lambda @f$,
    *      as @f$ \lambda @f$ is the coefficient of this layer's output
    *      @f$\ell_i@f$ in the overall Net loss
    *      @f$ E = \lambda_i \ell_i + \mbox{other loss terms}@f$; hence
    *      @f$ \frac{\partial E}{\partial \ell_i} = \lambda_i @f$.
    *      (*Assuming that this top Blob is not used as a bottom (input) by any
    *      other layer of the Net.)
    * @param bottom input Blob vector (length 2)
    *   -# @f$ (N \times C \times H \times W) @f$
    *      the predictions @f$\hat{y}@f$; Backward fills their diff with
    *      gradients @f$
    *        \frac{\partial E}{\partial \hat{y}} =
    *            \frac{1}{n} \sum\limits_{n=1}^N (\hat{y}_n - y_n)
    *      @f$ if propagate_down[0]
    *   -# @f$ (N) @f$
    *      the indices @f$ n \in [0, C \times H \times W - 1]@f$ of positive targets
    *
    * Gradients cannot be computed with respect to the index inputs (bottom[1]),
    * so this method ignores bottom[1] requires !propagate_down[1], crashing if they are set.
    */
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    int outer_num_, index_axis_;
    Blob<Dtype> diff_;
    Blob<Dtype> weights_;
    float threshold_;
    float null_scale_, positive_scale_;
};

}  // namespace caffe

#endif  // CAFFE_INDEXEDTHRESHOLD_LOSS_LAYER_HPP_
