// ------------------------------------------------------------------
// Jiang jia'nan .
// 2016/03/31
// ------------------------------------------------------------------
#ifndef CAFFE_BBOX_OBJ_LAYER_HPP_
#define CAFFE_BBOX_OBJ_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/FRCNN/util/frcnn_helper.hpp"

namespace caffe {

namespace Frcnn {

/*************************************************
BoxesObjLayer
Outputs object boxes after regression and nms,
and cls_indexes uses to specify classes which outputs,
-1 means output all classes

bottom: "rois"
bottom: "bbox_pred"
bottom: "cls_prob"
bottom: "im_info"
top: "boxes"
top: "scores"
type: "BoxesObj"
boxes_obj_param {
  cls_thresh: 0.7
  nms: 0.3
  box_vote:false
  cls_indexes: [-1]
}
**************************************************/
template <typename Dtype>
class BoxesObjLayer : public Layer<Dtype> {
 public:
  explicit BoxesObjLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){};

  virtual inline const char* type() const { return "BoxesObj"; }

  virtual inline int MinBottomBlobs() const { return 3; }
  virtual inline int MaxBottomBlobs() const { return 4; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }


 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    Dtype cls_thresh_;
    Dtype nms_;
    bool bbox_vote_;
    float bbox_vote_thresh;
    Dtype bbox_pred_;
    Dtype scores_;
    std::set<int> cls_index_;
    int box_num;
    int cls_num;

};

}  // namespace frcnn

}  // namespace caffe

#endif  // 
