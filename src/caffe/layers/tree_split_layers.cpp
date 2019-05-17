#include <vector>
#include <iostream>

#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TreeSplitLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  emptyfill_ = this->layer_param_.tree_split_param().emptyfill();
}

template<typename Dtype>
void TreeSplitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  for (int top_id = 0; top_id < top.size(); ++top_id) {
		top[top_id]->ReshapeLike(*bottom[0]);
  }
}

template<typename Dtype>
void TreeSplitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top) {
  inner_dim_ = bottom[0]->count() / bottom[0]->shape(0);
  thr1_ = bottom[1]->cpu_data();
  thr2_ = bottom[2]->cpu_data();
  const Dtype* samples = bottom[0]->cpu_data();
  const int count = bottom[1]->count();
  Dtype sum1_ = 0;
  Dtype sum2_ = 0;
  for (int i = 0; i < count; i++) {
  	sum1_ += thr1_[i];
    sum2_ += thr2_[i];
  }
  threshold1_ = sum1_ / count;
  threshold2_ = sum2_ / count;

  Dtype* left1  = top[0]->mutable_cpu_data();
  Dtype* left2 = top[1]->mutable_cpu_data();
  Dtype* right1  = top[2]->mutable_cpu_data();
  Dtype* right2 = top[3]->mutable_cpu_data();
  for (int top_id = 0; top_id < top.size(); top_id++) {
    caffe_set(top[top_id]->count(), emptyfill_, top[top_id]->mutable_cpu_data());
  }
  for (int index = 0; index < count; ++index) {
  	int mycount = inner_dim_*index;
	if (thr1_[index] <= threshold1_ && thr2_[index] <= threshold2_) {
		for (int i = 0; i < inner_dim_; ++i) {
			left1[mycount+i] = samples[mycount+i];
		}
	} else if (thr1_[index] <= threshold1_ && thr2_[index] > threshold2_) {
		for (int i = 0; i < inner_dim_; ++i) {
			left2[mycount+i] = samples[mycount+i];
		}
	} else if (thr1_[index] > threshold1_ && thr2_[index] <= threshold2_) {
    for (int i = 0; i < inner_dim_; ++i) {
      right1[mycount+i] = samples[mycount+i];
    }
  } else if (thr1_[index] > threshold1_ && thr2_[index] > threshold2_) {
    for (int i = 0; i < inner_dim_; ++i) {
      right2[mycount+i] = samples[mycount+i];
    }
  }
  }
} 

template<typename Dtype>
void TreeSplitLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff;
  Dtype* bot1_diff = bottom[0]->mutable_cpu_diff();

  int idx = 0;
  int mycount = 0; 
  thr1_ = bottom[1]->cpu_data();
  thr2_ = bottom[2]->cpu_data();
  for (int index = 0; index < bottom[1]->count(); ++index) {
  	mycount = inner_dim_*index;

    if (thr1_[index] <= threshold1_ && thr2_[index] <= threshold2_) {
      idx = 0;
    } else if (thr1_[index] <= threshold1_ && thr2_[index] > threshold2_) {
      idx = 1;
    } else if (thr1_[index] > threshold1_ && thr2_[index] <= threshold2_) {
      idx = 2;
    } else if (thr1_[index] > threshold1_ && thr2_[index] > threshold2_) {
      idx = 3;
    }

	  top_diff = top[idx]->cpu_diff();
	  for (int i = 0; i < inner_dim_; i++) {
		  bot1_diff[mycount+i] = top_diff[mycount+i];
	  }
  }
}

#ifdef CPU_ONLY
STUB_GPU(TreeSplitLayer);
#endif

INSTANTIATE_CLASS(TreeSplitLayer);
REGISTER_LAYER_CLASS(TreeSplit);

}  // namespace caffe
