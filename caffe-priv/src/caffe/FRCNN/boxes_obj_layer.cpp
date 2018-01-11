#include "caffe/FRCNN/boxes_obj_layer.hpp"
#include <iostream>
#include<stdio.h>
#include <iomanip>

namespace caffe {

namespace Frcnn {

using std::vector;

template <typename Dtype>
void BoxesObjLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
  const vector<Blob<Dtype> *> &top) {
  BoxesObjParameter _boxes_obj_param = this->layer_param_.boxes_obj_param();
  cls_thresh_ = _boxes_obj_param.cls_thresh();
  nms_ = _boxes_obj_param.nms();
  bbox_vote_ = _boxes_obj_param.box_vote();
  bbox_vote_thresh = 0.5;
  for (int c = 0; c < _boxes_obj_param.cls_indexes_size(); ++c){
    std::cout<< _boxes_obj_param.cls_indexes(c) << "_boxes_obj_param.cls_indexes(c)"<<std::endl;
    cls_index_.insert(_boxes_obj_param.cls_indexes(c));}

  top[0]->Reshape(1, 5, 1, 1);
  if (top.size() > 1) {
    top[1]->Reshape(1, 1, 1, 1);
  }
}

template <typename Dtype>
void BoxesObjLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                            const vector<Blob<Dtype> *> &top) {
  DLOG(ERROR) << "========== enter proposal layer";
  const Dtype *rois_ = bottom[0]->cpu_data();
  const Dtype *bbox_pred_ = bottom[1]->cpu_data();
  const Dtype *scores_ = bottom[2]->cpu_data();
  const Dtype *im_info = bottom[3]->cpu_data();
  float scale_factor = im_info[2];
  float width = im_info[1];
  float height = im_info[0];
  int box_num = bottom[0]->num();
  int cls_num = bottom[1]->channels()/4;
  std::vector<BBox<float> > results;
  std::vector<float> obj_scores;
  std::vector<Point4f<float> > boxes_obj;
  results.clear();
  for (int cls = 1; cls < cls_num; cls++){
      vector<BBox<float> > bbox;
      for (int i = 0; i < box_num; i++) {
        float score = scores_[i * cls_num + cls];
        Point4f<float> roi(rois_[(i * 5) + 1]/scale_factor, rois_[(i * 5) + 2]/scale_factor,
                           rois_[(i * 5) + 3]/scale_factor, rois_[(i * 5) + 4]/scale_factor);
        Point4f<float> delta(bbox_pred_[(i * cls_num + cls) * 4 + 0],
                             bbox_pred_[(i * cls_num + cls) * 4 + 1],
                             bbox_pred_[(i * cls_num + cls) * 4 + 2],
                             bbox_pred_[(i * cls_num + cls) * 4 + 3]);
        Point4f<float> box = caffe::Frcnn::bbox_transform_inv(roi, delta);
        box[0] = std::max(0.0f, box[0]);
        box[1] = std::max(0.0f, box[1]);
        box[2] = std::min(width/scale_factor-1.f, box[2]);
        box[3] = std::min(height/scale_factor-1.f, box[3]);
        bbox.push_back(BBox<float>(box, score, cls));
      }
      sort(bbox.begin(), bbox.end());
      vector<bool> select(box_num, true);
      // Apply NMS
      vector<BBox<float> > bbox_voting_list;
      for (int i = 0; i < box_num; i++)
        if (select[i]) {
          if (bbox[i].confidence < cls_thresh_) break;
          for (int j = i + 1; j < box_num; j++) {
            if (select[j]) {
              if (get_iou(bbox[i], bbox[j]) > nms_) {
                select[j] = false;
                }
              }
            }
          if (cls_index_.count(-1) != 1){
             if(cls_index_.count(bbox[i].id) != 1) continue;
            }
          results.push_back(bbox[i]);
          Point4f<float> final_bbox(bbox[i][0], bbox[i][1], bbox[i][2], bbox[i][3]);
          boxes_obj.push_back(final_bbox);
          obj_scores.push_back(bbox[i].confidence);
        }
    }
    top[0]->Reshape(boxes_obj.size(), 5, 1, 1);
    Dtype *top_data = top[0]->mutable_cpu_data();
    CHECK_EQ(boxes_obj.size(), obj_scores.size());
    for (size_t i = 0; i < boxes_obj.size(); i++) {
        Point4f<float> &box = boxes_obj[i];
        top_data[i * 5] = 0;
        for (int j = 1; j < 5; j++) {
            top_data[i * 5 + j] = box[j-1];
        }
    }
    if (top.size() > 1) {
        top[1]->Reshape(boxes_obj.size(), 1, 1, 1);
        for (size_t i = 0; i < boxes_obj.size(); i++) {
            top[1]->mutable_cpu_data()[i] = obj_scores[i];
        }
    }
}

template <typename Dtype>
void BoxesObjLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
    const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {
  for (int i = 0; i < propagate_down.size(); ++i) {
    if (propagate_down[i]) {
      NOT_IMPLEMENTED;
    }
  }
}

INSTANTIATE_CLASS(BoxesObjLayer);
REGISTER_LAYER_CLASS(BoxesObj);

} // namespace frcnn

} // namespace caffe
