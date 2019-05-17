#include <vector>
#include <iostream>

#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// this layer is opposite to batch_split_layer
// bottom[0] is groundtruth label or estimated label, used for batch merge
// bottom[1],... bottom[n] are the bottoms that need to be merged. 
// top[0] is the merged data

template<typename Dtype>
void BatchMergeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  const BatchMergeParameter& batch_merge_param = this->layer_param_.batch_merge_param();
    merge_threshold_.clear();
    std::copy(batch_merge_param.merge_threshold().begin(),
      batch_merge_param.merge_threshold().end(),
      std::back_inserter(merge_threshold_));
    emptyfill_ = batch_merge_param.emptyfill();
    backward_ = batch_merge_param.backward();

    const LayerParameter& layer_param_ = this->layer_param_;
    if (layer_param_.phase() == TRAIN) {
      train_ = true;
    } else {
      train_ = false;
    }
    int temp = bottom.size() - 2;
    CHECK_EQ(merge_threshold_.size(), temp);
}

template<typename Dtype>
void BatchMergeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
    top[0]->ReshapeLike(*bottom[1]);

    if (bottom.size() == 2) {
      top[0]->ShareData(*bottom[1]);
      top[0]->ShareDiff(*bottom[1]);
    }
}

template<typename Dtype>
void BatchMergeLayer<Dtype>::BackwardCopydiff(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top, int index, int idx) {
    int mycount = inner_dim_*index;
    const Dtype* diffs = top[0]->cpu_diff();
    if (idx == -1) {  // ignore label
      for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
        Dtype* out  = bottom[bottom_id]->mutable_cpu_diff();
        for (int i = 0; i < inner_dim_; ++i) {
          out[mycount+i] = 0;
        }
      }
    } else {
     for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
      Dtype* out  = bottom[bottom_id]->mutable_cpu_diff();
      if (bottom_id == idx) {
      // copy the original diff
       for (int i = 0; i < inner_dim_; ++i) {
          out[mycount+i] = diffs[mycount+i];
       }
      } else {
       for (int i = 0; i < inner_dim_; ++i) {
          out[mycount+i] = 0;
       }
      }
    }
  }
}

template<typename Dtype>
void BatchMergeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
    inner_dim_ = bottom[1]->count() / bottom[1]->shape(0);
    const Dtype* singal = bottom[0]->cpu_data();
    label_.clear();
    if (train_) {
    for (int i = 0; i < bottom[0]->count(); ++i) {
      label_.push_back(singal[i]);
    }
    } else {
    // testing: estimate the label  
      Dtype max_prob;
      int max_idx;
      int nclass = bottom[0]->count() / bottom[0]->shape(0);
      for (int i = 0; i < bottom[0]->shape(0); ++i) {
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

    int bottom_id = 0;
    const Dtype* bottom_data;
    Dtype* top_data = top[0]->mutable_cpu_data();
    for (int sample_id = 0; sample_id < label_.size(); ++sample_id) {
      int mycount = inner_dim_*sample_id;
      if (label_[sample_id] == -1) {
        for (int i = 0; i < inner_dim_; i++) {
          top_data[mycount+i] = emptyfill_;
        }
      } else {
        for (int i = 0; i < merge_threshold_.size(); i++) {
          if (label_[sample_id] <= merge_threshold_[i]) {
            bottom_id = i + 1;
            break;
          } else {
            bottom_id = i + 2;
          }
        }
      
        bottom_data = bottom[bottom_id]->cpu_data();
        for (int i = 0; i < inner_dim_; i++) {
          top_data[mycount+i] = bottom_data[mycount+i];
        }
      }
    }

/*    int top_id = 0;
    for (int sample_id = 0; sample_id < label_.size(); ++sample_id) {
      for (int i = 0; i < merge_threshold_.size(); i++) {
        if (label_[sample_id] == -1) { 
          top_id = -1; // ignore label
          break;
        } else if (label_[sample_id] <= merge_threshold_[i]) {
          top_id = i;
          break;
        } else {
          top_id = i + 1;
        }
      }
    this->ForwardCopydata(bottom, top, sample_id, top_id);
    } */
}

template<typename Dtype>
void BatchMergeLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if (backward_) {
      int bottom_id = 0;
      for (int sample_id = 0; sample_id < label_.size(); ++sample_id) {
        if (label_[sample_id] == -1) {
          bottom_id = -1;
        } else {
          for (int i = 0; i < merge_threshold_.size(); i++) {
            if (label_[sample_id] <= merge_threshold_[i]) {
              bottom_id = i + 1;
              break;
            } else {
              bottom_id = i + 2;
            }
          }
        }
        this->BackwardCopydiff(bottom, top, sample_id, bottom_id); 
//      mycount = inner_dim_*sample_id;
//      bot_diff = bottom[bottom_id]->mutable_cpu_diff();
//      for (int i = 0; i < inner_dim_; i++) {
//        bot_diff[mycount+i] = top_diff[mycount+i];
//      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(BatchMergeLayer);
#endif

INSTANTIATE_CLASS(BatchMergeLayer);
REGISTER_LAYER_CLASS(BatchMerge);

}  // namespace caffe
