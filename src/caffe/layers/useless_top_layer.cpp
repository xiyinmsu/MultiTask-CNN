#include <vector>
#include <iostream>

#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template<typename Dtype>
void UselessTopLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  vector<int> sz;
  sz.push_back(1);
  top[0]->Reshape(sz);
}

template<typename Dtype>
void UselessTopLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  top[0]->mutable_cpu_data()[0] = 0;
}

template<typename Dtype>
void UselessTopLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
}

#ifdef CPU_ONLY
STUB_GPU(UselessTopLayer);
#endif

INSTANTIATE_CLASS(UselessTopLayer);
REGISTER_LAYER_CLASS(UselessTop);

}  // namespace caffe
