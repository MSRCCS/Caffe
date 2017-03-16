#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/multi_accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void MultiAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void MultiAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  const Blob<Dtype>* prob = bottom[0];		//probabilities, dim = N * class_num  
  const Blob<Dtype>* label = bottom[1];	//labels, dim = N * Class_num
  
  Dtype accuracy = 0;
  //from pro
  CHECK_EQ(prob->count(), label->count());
  const Dtype* bottom_data = prob->cpu_data();
  const Dtype* bottom_label = label->cpu_data();
  int num = prob->num();
  int dim = prob->count() / prob->num();

  for (int i = 0; i < num; ++i) {
	  // Multi-label Accuracy

	  const Dtype* cur_bottom_data = bottom_data + i * dim;
	  const Dtype* cur_bottom_label = bottom_label + i * dim;

	  std::vector<std::pair<Dtype, int> > score(dim);
	  for (int j = 0; j < dim; j++) {
		  score[j] = std::make_pair(cur_bottom_data[j], j);
	  }
	  sort(score.begin(), score.end());

	  std::set<int> labels;
	  for (int j = 0; j < dim; j++) {
		  if (cur_bottom_label[j] > 0) labels.insert(j);
	  }

	  CHECK_GT(labels.size(), 0);
	  int acc = 0;
	  for (int j = dim - 1; j >= dim - (int)labels.size(); j--) {
		  if (labels.count(score[j].second)) acc++;
	  }

	  accuracy += (Dtype)acc / labels.size();
  }
  // LOG(INFO) << "Multi Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = accuracy/num;

  // Multi Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(MultiAccuracyLayer);
REGISTER_LAYER_CLASS(MultiAccuracy);

}  // namespace caffe
