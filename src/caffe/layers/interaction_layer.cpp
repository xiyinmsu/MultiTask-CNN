#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InteractionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.interaction_param().num_output();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.interaction_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
      this->blobs_.resize(1);
  }
  // Intialize the weight
  vector<int> weight_shape(2);
  weight_shape[0] = K_;
  weight_shape[1] = K_;
  this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
  // fill the weights
  shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.interaction_param().weight_filler()));
  weight_filler->Fill(this->blobs_[0].get());

  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void InteractionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.interaction_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);

  // size: 1*N_
  vector<int> shape(1, N_);   
  interact_multiplier_.Reshape(shape);
  caffe_set(N_, Dtype(1), interact_multiplier_.mutable_cpu_data());

  // size: M_*K_ 
  shape.clear();
  shape.push_back(M_);
  shape.push_back(K_);
  interact_multiplier_1_.Reshape(shape);
  caffe_set(M_*K_, Dtype(0), interact_multiplier_1_.mutable_cpu_data());

  // size: M_*M_
  shape.clear();
  shape.push_back(M_);
  shape.push_back(M_);
  interact_multiplier_2_.Reshape(shape);
  caffe_set(M_*M_, Dtype(0), interact_multiplier_2_.mutable_cpu_data());

  // size: 1*M_
  shape.clear();
  shape.push_back(M_);
  interact_multiplier_3_.Reshape(shape);
  caffe_set(M_, Dtype(0), interact_multiplier_3_.mutable_cpu_data());

  // size: 1*K_
  shape.clear();
  shape.push_back(K_);
  bottom_data_sample_.Reshape(shape);
  caffe_set(K_, Dtype(0), bottom_data_sample_.mutable_cpu_data());
}

template <typename Dtype>
void InteractionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();

  // X * Q
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., interact_multiplier_1_.mutable_cpu_data());

  // X * Q * X^T
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, M_, K_, (Dtype)1.,
      interact_multiplier_1_.cpu_data(), bottom_data, (Dtype)0., interact_multiplier_2_.mutable_cpu_data());

  // select diagonal matrix into a vector
  int incX = M_ + 1;
  int incY = 1;
  caffe_cpu_axpby<Dtype>(M_, (Dtype)1., interact_multiplier_2_.cpu_data(), incX, 
    (Dtype)0., interact_multiplier_3_.mutable_cpu_data(), incY);

/*  for (int i = 0; i < M_; i++) {
    interact_multiplier_3_.mutable_cpu_data()[i] = interact_multiplier_2_.cpu_data()[i*M_+i];
  } */

  // add diag(X*Q*X^T) to top_data
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        interact_multiplier_3_.cpu_data(),
        interact_multiplier_.cpu_data(), (Dtype)0., top_data);
}

template <typename Dtype>
void InteractionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  
  // Gradient with respect to weight
  if (this->param_propagate_down_[0]) {   
    caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);

    Dtype alpha = 1. / N_; 
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, M_, 1, N_, (Dtype)alpha, top_diff, interact_multiplier_.cpu_data(), 
      (Dtype)0., interact_multiplier_3_.mutable_cpu_data()); 

    for (int m = 0; m < M_; m++) {
      for (int k = 0; k < K_; k++) {
        bottom_data_sample_.mutable_cpu_data()[k] = bottom_data[m*K_+k];
      }
      alpha = interact_multiplier_3_.cpu_data()[m];
      caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, K_, K_, 1, alpha, 
        bottom_data_sample_.cpu_data(), bottom_data_sample_.cpu_data(), (Dtype)1., weight_diff);
    }
  }
  if (propagate_down[0]) {
    // dy / dx = X(Q + Q^T)
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, M_, K_, K_, (Dtype)1., 
        bottom_data, weight, (Dtype)0., bottom_diff);

    caffe_cpu_gemm(CblasTrans, CblasNoTrans, M_, K_, K_, (Dtype)1., 
        bottom_data, weight, (Dtype)1., bottom_diff);

    Dtype alpha = 1. / N_; 
    caffe_cpu_gemm(CblasNoTrans, CblasTrans, M_, N_, 1, (Dtype)alpha, top_diff, interact_multiplier_.cpu_data(), 
      (Dtype)0., interact_multiplier_3_.mutable_cpu_data()); 
    caffe_set(K_, (Dtype)1., bottom_data_sample_.mutable_cpu_data());
    caffe_cpu_gemm(CblasNoTrans, CblasTrans, M_, K_, 1, (Dtype)1., interact_multiplier_3_.cpu_data(), 
      bottom_data_sample_.cpu_data(), (Dtype)0., interact_multiplier_1_.mutable_cpu_data());

    // somehow the dot product does not work.... 
//    caffe_cpu_dot(bottom[0]->count(), interact_multiplier_1_.cpu_data(), bottom_diff);

    for (int i = 0; i < bottom[0]->count(); i++) {
      bottom_diff[i] = bottom_diff[i] * interact_multiplier_1_.cpu_data()[i];
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(InteractionLayer);
#endif

INSTANTIATE_CLASS(InteractionLayer);
REGISTER_LAYER_CLASS(Interaction);

}  // namespace caffe
