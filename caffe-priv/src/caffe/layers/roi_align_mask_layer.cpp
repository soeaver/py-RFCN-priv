#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/roi_align_mask_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void ROIAlignMaskLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ROIAlignMaskParameter roi_align_mask_param = this->layer_param_.roi_align_mask_param();
  CHECK_GT(roi_align_mask_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(roi_align_mask_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = roi_align_mask_param.pooled_h();
  pooled_width_ = roi_align_mask_param.pooled_w();
  spatial_scale_ = roi_align_mask_param.spatial_scale();
  spatial_shift_ = roi_align_mask_param.spatial_shift();
  half_part_ = roi_align_mask_param.half_part();
  roi_scale_ = roi_align_mask_param.roi_scale();
  mask_scale_ = roi_align_mask_param.mask_scale();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void ROIAlignMaskLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
  max_idx_.Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
  max_idy_.Reshape(bottom[1]->num(), channels_, pooled_height_,
      pooled_width_);
}

template <typename Dtype>
void ROIAlignMaskLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;

}

template <typename Dtype>
void ROIAlignMaskLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;

}


#ifdef CPU_ONLY
STUB_GPU(ROIAlignMaskLayer);
#endif

INSTANTIATE_CLASS(ROIAlignMaskLayer);
REGISTER_LAYER_CLASS(ROIAlignMask);

}  // namespace caffe
