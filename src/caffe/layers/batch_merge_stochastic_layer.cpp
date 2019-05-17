#include <vector>
#include <iostream>

#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// merge the probablity of each batch based on the estimated pose probablity
// bottom[0] is prob_pos
// bottom[1] is prob_batch1
// bottom[2] is prob_batch2
// minimum of three bottoms

template<typename Dtype>
void BatchMergeStochasticLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  const BatchMergeStochasticParameter& batch_merge_stochastic_param = this->layer_param_.batch_merge_stochastic_param();
    merge_threshold_.clear();
    std::copy(batch_merge_stochastic_param.merge_threshold().begin(),
      batch_merge_stochastic_param.merge_threshold().end(),
      std::back_inserter(merge_threshold_));

    nSample_ = bottom[1]->shape(0);   
    nGroup_ = merge_threshold_.size()+1;
    CHECK_EQ(nGroup_, bottom.size()-1);

    this->blobs_.resize(3);
    vector<int> sz;
    sz.push_back(nSample_);
    for (int i = 0; i < nGroup_; i++) {
      this->blobs_[i].reset(new Blob<Dtype>(sz));
        caffe_set(this->blobs_[i]->count(), Dtype(0),
                this->blobs_[i]->mutable_cpu_data());
    }
}

template<typename Dtype>
void BatchMergeStochasticLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
    top[0]->ReshapeLike(*bottom[1]);
    vector<int> sz;
    channels_ = bottom[1]->count() / bottom[1]->shape(0);
    sz.push_back(channels_);

    batch_sum_multiplier_.Reshape(sz);
    caffe_set(batch_sum_multiplier_.count(), Dtype(1),
        batch_sum_multiplier_.mutable_cpu_data());
    prob_multiplier_.ReshapeLike(*bottom[1]);
    caffe_set(prob_multiplier_.count(), Dtype(0),
        prob_multiplier_.mutable_cpu_data());
}


template<typename Dtype>
void BatchMergeStochasticLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
    inner_dim_ = bottom[1]->count() / bottom[1]->shape(0);   
    const Dtype* prob_pos = bottom[0]->cpu_data();

    // accumulate the probability of each pose group
    nClass_ = bottom[0]->count() / bottom[0]->shape(0);
    for (int i = 0; i < nGroup_; i++) {
      caffe_set(this->blobs_[i]->count(), Dtype(0), this->blobs_[i]->mutable_cpu_data());
    }
    Dtype p;
    int g = 0;
    for (int i = 0; i < nSample_; ++i) {
      for (int j = 0; j < nClass_; ++j) {
        p = prob_pos[nClass_*i+j];
        for (int k = 0; k < nGroup_-1; k++) {
          if ( j <= merge_threshold_[k] ) {
            g = k;
            break;
          } else {
            g = k+1;
          }
        }
        this->blobs_[g]->mutable_cpu_data()[i] = this->blobs_[g]->mutable_cpu_data()[i] + p;
      }
    }

    // stochastic combination for top[0]
    Dtype* top_data = top[0]->mutable_cpu_data();
    caffe_set(top[0]->count(), Dtype(0), top_data);
    for (int i = 0; i < nGroup_; ++i) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, nSample_, channels_, 1, 1.,
        this->blobs_[i]->cpu_data(), batch_sum_multiplier_.cpu_data(), 0., prob_multiplier_.mutable_cpu_data());
      caffe_mul<Dtype>(top[0]->count(), prob_multiplier_.cpu_data(), bottom[i+1]->cpu_data(), prob_multiplier_.mutable_cpu_data());
      caffe_add<Dtype>(top[0]->count(), prob_multiplier_.cpu_data(), top[0]->cpu_data(), top_data);
    }
}

template<typename Dtype>
void BatchMergeStochasticLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  const Dtype* top_diff = top[0]->cpu_diff();
  for (int i = 0; i < nGroup_; ++i) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, nSample_, channels_, 1, 1.,
      this->blobs_[i]->cpu_data(), batch_sum_multiplier_.cpu_data(), 0., prob_multiplier_.mutable_cpu_diff());
    caffe_mul<Dtype>(bottom[i+1]->count(), top_diff, prob_multiplier_.cpu_diff(), bottom[i+1]->mutable_cpu_diff());

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, nSample_, 1, 1, 1., 
      bottom[i+1]->cpu_data(), batch_sum_multiplier_.cpu_data(), 0., this->blobs_[i]->mutable_cpu_data());
  }

/*  Dtype* bot_diff = bottom[0]->mutable_cpu_diff();
  int g = 0;
  for (int i = 0; i < nSample_; ++i) {
    for (int j = 0; j < nClass_; ++j) {
      for (int k = 0; k < nGroup_-1; ++k) {
        if (j <= merge_threshold_[k]) {
          g = k;
        } else {
          g = k + 1;
        }
      }
      Dtype diff = this->blobs_[g]->cpu_data()[i];
      bot_diff[i*nClass_+j] = diff;
    }
  } */
}

#ifdef CPU_ONLY
STUB_GPU(BatchMergeStochasticLayer);
#endif

INSTANTIATE_CLASS(BatchMergeStochasticLayer);
REGISTER_LAYER_CLASS(BatchMergeStochastic);

}  // namespace caffe
