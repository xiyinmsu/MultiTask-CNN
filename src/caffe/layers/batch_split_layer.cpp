#include <vector>
#include <iostream>

#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// this layer needs two bottoms:
// bottom[0] is the batch features or labels
// bottom[1] is the singal used for splitting
	// during training, we use the ground truth label for splitting the batch samples into multiple blobs
	// during testing, we use th estimated label, usually it is a vector for each sample, we find the label inside this layer
// the user has to specify all thresholds used for the splitting 
// all tops are the same size as bottom[0], with a subset of the batch samples selected, others are filled with a value specified by the user. 

template<typename Dtype>
void BatchSplitLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
	const BatchSplitParameter& batch_split_param = this->layer_param_.batch_split_param();
   	split_threshold_.clear();
   	std::copy(batch_split_param.split_threshold().begin(),
      batch_split_param.split_threshold().end(),
      std::back_inserter(split_threshold_));
   	emptyfill_ = batch_split_param.emptyfill();
   	backward_ = batch_split_param.backward();

   	const LayerParameter& layer_param_ = this->layer_param_;
   	if (layer_param_.phase() == TRAIN) {
   		train_ = true;
   	} else {
   		train_ = false;
   	}
   	int temp = top.size() - 1;
   	CHECK_EQ(split_threshold_.size(), temp);
}

template<typename Dtype>
void BatchSplitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
   	for (int top_id = 0; top_id < top.size(); ++top_id) {
		top[top_id]->ReshapeLike(*bottom[0]);
   	}
  	if (top.size() == 1) {
    	top[0]->ShareData(*bottom[0]);
    	top[0]->ShareDiff(*bottom[0]);
  	}
}

template<typename Dtype>
void BatchSplitLayer<Dtype>::ForwardCopydata(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top, int index, int idx) {
  	int mycount = inner_dim_*index;
  	const Dtype* samples = bottom[0]->cpu_data();
    if (idx == -1) {  // ignore label
      for (int top_id = 0; top_id < top.size(); ++top_id) {
        Dtype* out  = top[top_id]->mutable_cpu_data();
        for (int i = 0; i < inner_dim_; ++i) {
          out[mycount+i] = emptyfill_;
        }
      }
    } else {
  	 for (int top_id = 0; top_id < top.size(); ++top_id) {
		  Dtype* out  = top[top_id]->mutable_cpu_data();
		  if (top_id == idx) {
		  // copy the original data
			 for (int i = 0; i < inner_dim_; ++i) {
				  out[mycount+i] = samples[mycount+i];
			 }
		  } else {
		  // fill with emptyfill_ for samples 
			 for (int i = 0; i < inner_dim_; ++i) {
				  out[mycount+i] = emptyfill_;
			 }
		  }
  	}
  }
}

template<typename Dtype>
void BatchSplitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  	inner_dim_ = bottom[0]->count() / bottom[0]->shape(0);
  	const Dtype* singal = bottom[1]->cpu_data();
  	label_.clear();
  	if (train_) {
		for (int i = 0; i < bottom[1]->count(); ++i) {
			label_.push_back(singal[i]);
		}
   	} else {
		// testing: estimate the label 	
		Dtype max_prob;
		int max_idx;
		int nclass = bottom[1]->count() / bottom[1]->shape(0);
		for (int i = 0; i < bottom[1]->shape(0); ++i) {
			max_prob = singal[nclass*i];
			max_idx = 0;
			for (int j = 1; j < nclass; ++j) {
				if (singal[nclass*i+j] > max_prob) {
					max_prob = singal[nclass*i+j];
					max_idx = j;
				}
			}
			label_.push_back(max_idx);
		}
	}

  	int top_id = 0;
  	for (int sample_id = 0; sample_id < label_.size(); ++sample_id) {
  		for (int i = 0; i < split_threshold_.size(); i++) {
  			if (label_[sample_id] == -1) { 
          top_id = -1; // ignore label
          break;
        } else if (label_[sample_id] <= split_threshold_[i]) {
  				top_id = i;
  				break;
  			} else {
  				top_id = i + 1;
  			}
  		}
		this->ForwardCopydata(bottom, top, sample_id, top_id);
  	}
}

template<typename Dtype>
void BatchSplitLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

	if (backward_) {
		const Dtype* top_diff;
  	Dtype* bot_diff = bottom[0]->mutable_cpu_diff();
  	int mycount = 0; 
  	int top_id = 0;
  	for (int sample_id = 0; sample_id < label_.size(); ++sample_id) {
      mycount = inner_dim_*sample_id;
      if (label_[sample_id] == -1) {
        for (int i = 0; i < inner_dim_; i++) {
          bot_diff[mycount+i] = 0;
        }
      } else {
        for (int i = 0; i < split_threshold_.size(); i++) {
          if (label_[sample_id] <= split_threshold_[i]) {
            top_id = i;
            break;
          } else {
            top_id = i + 1;
          }
        }
       
       top_diff = top[top_id]->cpu_diff();
       for (int i = 0; i < inner_dim_; i++) {
        bot_diff[mycount+i] = top_diff[mycount+i];
       }
      } 
    }
	}
}

#ifdef CPU_ONLY
STUB_GPU(BatchSplitLayer);
#endif

INSTANTIATE_CLASS(BatchSplitLayer);
REGISTER_LAYER_CLASS(BatchSplit);

}  // namespace caffe
