#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InferenceProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inference_product_param().num_output();
  bias_term_ = this->layer_param_.inference_product_param().bias_term();
  infer_weight_ = this->layer_param_.inference_product_param().infer_weight();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inference_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(3);
    } else {
      this->blobs_.resize(2);
    }
    // Intialize the weight (N_*K_)
    vector<int> weight_shape(2);
    weight_shape[0] = N_;
    weight_shape[1] = K_;
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inference_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());

    // Intialize the inference matrix Q (K_*K_)
    weight_shape[0] = K_;
    this->blobs_[1].reset(new Blob<Dtype>(weight_shape));
    // fill the inference matrix 
    shared_ptr<Filler<Dtype> > inference_weight_filler(GetFiller<Dtype>(
        this->layer_param_.inference_product_param().inference_weight_filler()));
    inference_weight_filler->Fill(this->blobs_[1].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[2].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inference_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[2].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void InferenceProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inference_product_param().axis());
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
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }

  vector<int> shape(1, N_);
  inference_multiplier_.Reshape(shape);
  caffe_set(N_, Dtype(1), inference_multiplier_.mutable_cpu_data());

  shape.clear();
  shape.push_back(M_);
  shape.push_back(K_);
  inference_multiplier_1_.Reshape(shape);
  caffe_set(M_*K_, Dtype(0), inference_multiplier_1_.mutable_cpu_data());

  shape.clear();
  shape.push_back(M_);
  shape.push_back(M_);
  inference_multiplier_2_.Reshape(shape);
  caffe_set(M_*M_, Dtype(0), inference_multiplier_2_.mutable_cpu_data());

  shape.clear();
  shape.push_back(M_);
  inference_multiplier_3_.Reshape(shape);
  caffe_set(M_, Dtype(0), inference_multiplier_3_.mutable_cpu_data());

  shape.clear();
  shape.push_back(K_);
  bottom_data_sample_.Reshape(shape);
  caffe_set(K_, Dtype(0), bottom_data_sample_.mutable_cpu_data());
}

template <typename Dtype>
void InferenceProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* inference_weight = this->blobs_[1]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);

  // X * Q
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, K_, (Dtype)1.,
      bottom_data, inference_weight, (Dtype)0., inference_multiplier_1_.mutable_cpu_data());

  // X * Q * X^T
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, M_, K_, (Dtype)1.,
      inference_multiplier_1_.cpu_data(), bottom_data, (Dtype)0., inference_multiplier_2_.mutable_cpu_data());

  // select diagonal matrix into a vector
  for (int i = 0; i < M_; i++) {
    inference_multiplier_3_.mutable_cpu_data()[i] = inference_multiplier_2_.cpu_data()[i*M_+i] * infer_weight_;
  }

  // add diag(X*Q*X^T) to top_data
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        inference_multiplier_3_.cpu_data(),
        inference_multiplier_.cpu_data(), (Dtype)1., top_data);

  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[2]->cpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InferenceProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  
  if (this->param_propagate_down_[0]) {
    // Gradient with respect to weight
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
        top_diff, bottom_data, (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
  }

  if (this->param_propagate_down_[1]) {
    // Gradient with respect to reference_weight
    caffe_set(this->blobs_[1]->count(), Dtype(0), this->blobs_[1]->mutable_cpu_diff());
    Dtype top_diff_sum;
    Dtype alpha;
    for (int m = 0; m < M_; m++) {
      top_diff_sum = 0;
      for (int n = 0; n < N_; n++) {
        top_diff_sum += top_diff[m*N_+n];
      }
      for (int k = 0; k < K_; k++) {
        bottom_data_sample_.mutable_cpu_data()[k] = bottom_data[m*K_+k];
      }
      alpha = top_diff_sum / M_; 
      alpha = alpha*infer_weight_;
      caffe_cpu_gemv(CblasNoTrans, K_, 1, alpha, bottom_data_sample_.cpu_data(), 
        bottom_data_sample_.cpu_data(), (Dtype)1., this->blobs_[1]->mutable_cpu_diff());
    }

    // compute X*X^T
/*    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, K_, M_, (Dtype)1., 
        bottom_data, bottom_data, (Dtype)0., this->blobs_[1]->mutable_cpu_diff());
    // compute summation of top_diff
    Dtype sum = 0;
    for (int i = 0; i < top[0]->count(); i++) {
      sum += top_diff[i];
    }
    avg_ = sum / top[0]->count();
    avg_ = avg_*infer_weight_/M_;  // multiply by a scaler to make the values small
    caffe_scal(this->blobs_[1]->count(), avg_, this->blobs_[1]->mutable_cpu_diff());
*/  }
  if (bias_term_ && this->param_propagate_down_[2]) {
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[2]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    // Gradient with respect to bottom data
    // dy / dx = W + ZM^T*X*(Q + Q^T)
    // compute X*Q and X*Q^T respectively
    caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, M_, K_, K_, (Dtype)1., 
        bottom_data, this->blobs_[1]->cpu_data(), (Dtype)0., 
        inference_multiplier_1_.mutable_cpu_data());
    caffe_cpu_gemm(CblasNoTrans, CblasTrans, M_, K_, K_, (Dtype)1., 
        bottom_data, this->blobs_[1]->cpu_data(), (Dtype)1., 
        inference_multiplier_1_.mutable_cpu_data());
    caffe_copy(inference_multiplier_1_.count(), inference_multiplier_1_.cpu_data(), bottom[0]->mutable_cpu_diff());

    caffe_set(inference_multiplier_.count(), (Dtype)1., inference_multiplier_.mutable_cpu_data());
    Dtype alpha = 1 / N_; 
    caffe_cpu_gemv(CblasNoTrans, M_, N_, (Dtype)alpha, top_diff, inference_multiplier_.cpu_data(), 
      (Dtype)0., inference_multiplier_3_.mutable_cpu_data());
    
    caffe_set(bottom_data_sample_.count(), (Dtype)1., bottom_data_sample_.mutable_cpu_data());
    caffe_cpu_gemv(CblasNoTrans, M_, 1, (Dtype)1., inference_multiplier_3_.cpu_data(), 
      bottom_data_sample_.cpu_data(), (Dtype)0., inference_multiplier_1_.mutable_cpu_data());

    caffe_cpu_dot(bottom[0]->count(), inference_multiplier_1_.cpu_data(), 
      bottom[0]->mutable_cpu_diff());
    caffe_scal(bottom[0]->count(), (Dtype)infer_weight_, bottom[0]->mutable_cpu_diff());

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, this->blobs_[0]->cpu_data(), (Dtype)1.,
        bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(InferenceProductLayer);
#endif

INSTANTIATE_CLASS(InferenceProductLayer);
REGISTER_LAYER_CLASS(InferenceProduct);

}  // namespace caffe
