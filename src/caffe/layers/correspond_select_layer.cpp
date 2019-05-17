#include <vector>
#include <iostream>

#include "caffe/common_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// bottom[0]: data_gallery (frontal faces for all subjects)
// bottom[1]: label_id_gallery (identity for frontal faces)
// bottom[2]: label_id (identity labels for current batch)
// Goal: select a frontal face for each subject in current batch based on the identity labels

template <typename Dtype>
void CorrespondSelectLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  nGallery_ = bottom[1]->count();
  nSample_ = bottom[2]->count();
  inner_dim_ = bottom[0]->count() / nGallery_; 
}

template<typename Dtype>
void CorrespondSelectLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape = bottom[0]->shape();
  top_shape[0] = nSample_;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void CorrespondSelectLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* data_gallery = bottom[0]->cpu_data();
  const Dtype* id_gallery = bottom[1]->cpu_data();
  const Dtype* id_sample = bottom[2]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  Dtype id_g, id_s; 
  int start_g, start_s; 
  for (int s = 0; s < nSample_; s++) {
    id_s = id_sample[s];
    for (int g = 0; g < nGallery_; g++) {
      id_g = id_gallery[g];
      if (id_s == id_g) {
        start_s = s*inner_dim_; 
        start_g = g*inner_dim_; 
        for (int i = 0; i < inner_dim_; i++) {
          top_data[start_s+i] = data_gallery[start_g+i];
        }
        break;
      }
    }
  }
}

template <typename Dtype>
void CorrespondSelectLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_CLASS(CorrespondSelectLayer);
REGISTER_LAYER_CLASS(CorrespondSelect);

}  // namespace caffe