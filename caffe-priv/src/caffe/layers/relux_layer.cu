#include <algorithm>
#include <vector>

#include "caffe/layers/relux_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ReLUXForward(const int n, const Dtype* in, Dtype* out,
    Dtype negative_slope, Dtype maximal_value) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = min(max(Dtype(0), in[index]), maximal_value);
    // if (in[index] > Dtype(0) && in[index] < maximal_value) {
    //   out[index] = in[index];
    // } else if (in[index] <= Dtype(0)) {
    //   out[index] = in[index] * negative_slope;
    // } else {
    //   out[index] = maximal_value;
    // }
  }
}

template <typename Dtype>
void ReLUXLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relux_param().negative_slope();
  Dtype maximal_value = this->layer_param_.relux_param().maximal_value();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ReLUXForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, negative_slope, maximal_value);
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void ReLUXBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype negative_slope, Dtype maximal_value) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0 && in_data[index] < maximal_value)
        + (in_data[index] <= 0) * negative_slope);
  }
}

template <typename Dtype>
void ReLUXLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relux_param().negative_slope();
    Dtype maximal_value = this->layer_param_.relux_param().maximal_value();
    // NOLINT_NEXT_LINE(whitespace/operators)
    ReLUXBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, negative_slope, maximal_value);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ReLUXLayer);


}  // namespace caffe
