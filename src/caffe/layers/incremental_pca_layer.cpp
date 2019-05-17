#include <algorithm>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void IncrementalPCALayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  IncrementalPCAParameter param = this->layer_param_.incremental_pca_param();
  moving_average_fraction_ = param.moving_average_fraction();
  moving_covariance_fraction_ = param.moving_covariance_fraction();
  update_interval_ = param.update_interval();
  nbasis_ = param.nbasis();
  iter_count_ = 0;
  use_global_stats_ = this->phase_ == TEST;
  if (param.has_use_global_stats())
    use_global_stats_ = param.use_global_stats();
  if (bottom[0]->num_axes() == 1)
    channels_ = 1;
  else
    channels_ = bottom[0]->shape(1);

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(2+nbasis_);
    vector<int> sz;
    sz.push_back(channels_);
    this->blobs_[0].reset(new Blob<Dtype>(sz));  // mean_ vector
    sz[0] = channels_*channels_;
    this->blobs_[1].reset(new Blob<Dtype>(sz));  // covariance_matrix
    for (int i = 0; i < 2; ++i) {
      caffe_set(this->blobs_[i]->count(), Dtype(0),
                this->blobs_[i]->mutable_cpu_data());
    }

    sz[0] = channels_;
    Dtype value = 1. / sqrt(channels_);
    for (int n = 0; n < nbasis_; n++) {
    	this->blobs_[2+n].reset(new Blob<Dtype>(sz));     
      for (int i = 0; i < this->blobs_[2+n]->count(); i++) {
        float randvalue = (rand() % 100) / 1000.0;
        this->blobs_[2+n]->mutable_cpu_data()[i] = value + randvalue;
      }
 //   	caffe_set(this->blobs_[2+n]->count(), value + randvalue,
 //               this->blobs_[2+n]->mutable_cpu_data());
    }
  }
}

template <typename Dtype>
void IncrementalPCALayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom[0]->num_axes() >= 1)
    CHECK_EQ(bottom[0]->shape(1), channels_);

  int num = bottom[0]->shape(0);
  vector<int> sz;
  sz.push_back(num);
  for (int i = 0; i < top.size(); i++) {
  	top[i]->Reshape(sz);
  }

  sz[0] = channels_;
  batch_mean_.Reshape(sz);
  princomp_.Reshape(sz);
  caffe_set(batch_mean_.count(), Dtype(0),
        batch_mean_.mutable_cpu_data());
  caffe_set(princomp_.count(), Dtype(0),
        princomp_.mutable_cpu_data());

  sz[0] = channels_*channels_;
  covariance_.Reshape(sz);
  batch_covariance_.Reshape(sz);
  caffe_set(covariance_.count(), Dtype(0),
        covariance_.mutable_cpu_data());
  caffe_set(batch_covariance_.count(), Dtype(0),
        batch_covariance_.mutable_cpu_data());

  sz[0] = num;
  batch_sum_multiplier_.Reshape(sz);
  caffe_set(batch_sum_multiplier_.count(), Dtype(1),
        batch_sum_multiplier_.mutable_cpu_data());

  sz[0] = num*channels_;
  temp_covariance_.Reshape(sz);
  caffe_set(temp_covariance_.count(), Dtype(0),
        temp_covariance_.mutable_cpu_data());
}


template <typename Dtype>
void IncrementalPCALayer<Dtype>::EigenDecomposition(const Blob<Dtype>& covMatrix) {
	Eigen::MatrixXd covMax(channels_, channels_);
	int n = 0;
	for (int i = 0; i < channels_; i++) {
		for (int j = 0; j < channels_; j++) {
			covMax(i, j) = covMatrix.cpu_data()[n];
			n++;
		}
	}

	Eigen::EigenSolver<Eigen::MatrixXd> es(covMax);
	Eigen::MatrixXd V = es.pseudoEigenvectors();
	for (int i = 0; i < nbasis_; i++) {
		for (int j = 0; j < channels_; j++) {
			this->blobs_[2+i]->mutable_cpu_data()[j] = V(j,i);
		}
	}
 }

template <typename Dtype>
void IncrementalPCALayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  int num = bottom[0]->shape(0);

  if (!use_global_stats_) {
    iter_count_++;
//    std::cout << "iter_count = " << iter_count_ << std::endl;
    caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1./num,
        bottom_data, batch_sum_multiplier_.cpu_data(), 0.,
        batch_mean_.mutable_cpu_data());

    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1, 
        batch_sum_multiplier_.cpu_data(), batch_mean_.cpu_data(), 0., 
        temp_covariance_.mutable_cpu_data()); 

    caffe_sub(temp_covariance_.count(), bottom_data, temp_covariance_.cpu_data(), 
        temp_covariance_.mutable_cpu_data());

    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, channels_, num, 1./num, 
        temp_covariance_.cpu_data(), temp_covariance_.cpu_data(), 0., 
        batch_covariance_.mutable_cpu_data());

    // update mean_ and covariance_ 
    caffe_cpu_axpby(batch_mean_.count(), Dtype(1)-moving_average_fraction_, batch_mean_.cpu_data(),
        moving_average_fraction_, this->blobs_[0]->mutable_cpu_data());

    caffe_cpu_axpby(batch_covariance_.count(), Dtype(1)-moving_covariance_fraction_, batch_covariance_.cpu_data(),
        moving_covariance_fraction_, this->blobs_[1]->mutable_cpu_data());

    if (iter_count_ == update_interval_) {
      iter_count_ = 0;
      caffe_copy(covariance_.count(), this->blobs_[1]->cpu_data(), covariance_.mutable_cpu_data());
      this->EigenDecomposition(covariance_);
    }
  }

  // project bottom_data to nbasis_ dimension 
  for (int i = 0; i < top.size(); i++) {
  	  Dtype* top_data = top[i]->mutable_cpu_data();
  	  caffe_copy(princomp_.count(), this->blobs_[2+i]->cpu_data(), princomp_.mutable_cpu_data());
  	  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, channels_, 1.,
             bottom_data, princomp_.cpu_data(), 0, top_data);
  }
}

template <typename Dtype>
void IncrementalPCALayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  CHECK(!use_global_stats_);
//  NOT_IMPLEMENTED;
//  int num = bottom[0]->shape(0);
//  Dtype* top_diff = top[0]->mutable_cpu_diff();
/*  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  for (int i = 0; i < bottom[0]->count(); i++) {
    bottom_diff[i] = 0;
  }

  for (int i = 0; i <this->blobs_[0]->count(); i++) {
    this->blobs_[0]->mutable_cpu_diff()[i] = 0;
  }
  for (int i = 0; i <this->blobs_[1]->count(); i++) {
    this->blobs_[1]->mutable_cpu_diff()[i] = 0;
  }
  for (int i = 0; i <this->blobs_[2]->count(); i++) {
    this->blobs_[2]->mutable_cpu_diff()[i] = 0;
  } */
  
/*  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, 1, channels_, 1, 
        top_diff, this->blobs_[2]->cpu_data(), 0., 
        bottom_diff); */
}



#ifdef CPU_ONLY
STUB_GPU(IncrementalPCALayer);
#endif

INSTANTIATE_CLASS(IncrementalPCALayer);
REGISTER_LAYER_CLASS(IncrementalPCA);
}  // namespace caffe
