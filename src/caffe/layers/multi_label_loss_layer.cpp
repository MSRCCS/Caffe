#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/multi_label_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void MultiLabelLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	LossLayer<Dtype>::LayerSetUp(bottom, top);
	LossLayer<Dtype>::Reshape(bottom, top);
    CHECK_GE(bottom[0]->shape(1), bottom[1]->shape(1) * 2)
        << "Data must have twice dimensions than label.";
    // the prob output should have the same shape as data input / 2
    vector<int> prob_shape = bottom[0]->shape();
    prob_shape[1] /= 2;
    prob_pos_.Reshape(prob_shape);
	prob_neg_.Reshape(prob_shape);
	normalize_ = this->layer_param_.loss_param().normalize();
	if (top.size() >= 2)
		top[1]->Reshape(prob_shape); 
}

template <typename Dtype>
void UnrollLabel(const Dtype* compact_label, int compact_label_dim, vector<int>& label)
{
    memset(&label[0], 0, sizeof(int) * label.size());

    for (int i = 0; i < compact_label_dim; i++)
    {
        int value = static_cast<int>(compact_label[i]);
        if (value < 0)  // negative numbers are padding values
            break;
        CHECK_LT(value, label.size()) << "Label value too large: " << value << " vs. " << label.size();
        label[value] = 1;
    }
}
template void UnrollLabel<float>(const float* compact_label, int compact_label_dim, vector<int>& label);
template void UnrollLabel<double>(const double* compact_label, int compact_label_dim, vector<int>& label);

template <typename Dtype>
void MultiLabelLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	int num = bottom[0]->shape(0);
	int dim = bottom[0]->shape(1) / 2;
	const Dtype* data = bottom[0]->cpu_data();
	const Dtype* compact_label = bottom[1]->cpu_data();
    const int compact_dim = bottom[1]->shape(1);
    Dtype* prob_pos = prob_pos_.mutable_cpu_data();
	Dtype* prob_neg = prob_neg_.mutable_cpu_data();
	Dtype loss = 0;
    vector<int> label(dim); // allocate unrolled label space
    for (int n = 0; n < num; n++)
	{
        // unroll the compact form labels
        UnrollLabel(compact_label + compact_dim * n, compact_dim, label);
        // compute loss
		for (int k = 0; k < dim; k++)
		{
			const int label_value = label[k];
			DCHECK_GE(label_value, 0);
			DCHECK_LE(label_value, 1);
			Dtype x_pos = *data++;
			Dtype x_neg = *data++;
			Dtype x_max = std::max<Dtype>(x_pos, x_neg);
			Dtype x_pos_exp = exp(x_pos - x_max);
			Dtype x_neg_exp = exp(x_neg - x_max);
			Dtype prob_pos_value = x_pos_exp / (x_pos_exp + x_neg_exp);
			Dtype prob_neg_value = x_neg_exp / (x_pos_exp + x_neg_exp);
			*prob_pos++ = prob_pos_value;
			*prob_neg++ = prob_neg_value;
			if (label_value == 0)
				loss -= log(std::max(prob_neg_value, Dtype(FLT_MIN)));
			else
				loss -= log(std::max(prob_pos_value, Dtype(FLT_MIN)));
		}
	}

	if (normalize_)
		top[0]->mutable_cpu_data()[0] = loss / (num * dim);
	else
		top[0]->mutable_cpu_data()[0] = loss / num;

	if (top.size() == 2) {
		top[1]->ShareData(prob_pos_);
	}
}

template <typename Dtype>
void MultiLabelLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (propagate_down[1]) {
		LOG(FATAL) << this->type()
			<< " Layer cannot backpropagate to label inputs.";
	}
	if (propagate_down[0]) {
		int num = bottom[0]->shape(0);
		int dim = bottom[0]->shape(1) / 2;
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		const Dtype* compact_label = bottom[1]->cpu_data();
        const int compact_dim = bottom[1]->shape(1);
		const Dtype* prob_pos = prob_pos_.cpu_data();
		const Dtype* prob_neg = prob_neg_.cpu_data();
		Dtype* b_diff = bottom_diff;
        vector<int> label(dim); // allocate unrolled label space
		for (int n = 0; n < num; n++)
		{
            // unroll the compact form labels
            UnrollLabel(compact_label + compact_dim * n, compact_dim, label);
            // compute gradient
            for (int k = 0; k < dim; k++)
			{
				const int label_value = label[k];
				Dtype prob_pos_value = *prob_pos++;
				Dtype prob_neg_value = *prob_neg++;
				if (label_value == 0)
				{
					*b_diff++ = prob_pos_value;	// for x_pos
					*b_diff++ = -prob_pos_value;	// for x_neg
				}
				else
				{
					*b_diff++ = -prob_neg_value;	// for x_pos
					*b_diff++ = prob_neg_value;	// for x_neg
				}
			}
		}
		// Scale gradient
		const Dtype loss_weight = top[0]->cpu_diff()[0];
		if (normalize_)
			caffe_scal(bottom[0]->count(), loss_weight / (num * dim), bottom_diff);
		else
			caffe_scal(bottom[0]->count(), loss_weight / num, bottom_diff);
	}
}

#ifdef CPU_ONLY
STUB_GPU(MultiLabelLossLayer);
#endif

INSTANTIATE_CLASS(MultiLabelLossLayer);
REGISTER_LAYER_CLASS(MultiLabelLoss);

}  // namespace caffe
