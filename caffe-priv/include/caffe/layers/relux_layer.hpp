#ifndef CAFFE_RELUX_LAYER_HPP_
#define CAFFE_RELUX_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

/**
 * @brief Rectified Linear Unit non-linearity under X @f$ y = \min(max(0, x), X), x default 6 @f$.
 *        The simple min/max is fast to compute, and the function does not saturate.
 */
template <typename Dtype>
class ReLUXLayer : public NeuronLayer<Dtype> {
 public:
  /**
   * @param param provides ReLUPXarameter relux_param,
   *     with ReLUXLayer options:
   *   - negative_slope (\b optional, default 0).
   *     the value @f$ \nu @f$ by which negative values are multiplied.
   *   - maximal_value (\b optional, default 6 must >=0).
   */
  explicit ReLUXLayer(const LayerParameter& param)
      : NeuronLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "ReLUX"; }

 protected:
  /**
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$
   * @param top output Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the computed outputs @f$
   *        y = \min(max(0, x), X)
   *      @f$ by default.  If a non-zero negative_slope @f$ \nu @f$ is provided,
   *      the computed outputs are @f$ y = \min(max(0, x), X) + \nu \min(0, x) @f$.
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the error gradient w.r.t. the ReLUX inputs.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   *   -# @f$ (N \times C \times H \times W) @f$
   *      containing error gradients @f$ \frac{\partial E}{\partial y} @f$
   *      with respect to computed outputs @f$ y @f$
   * @param propagate_down see Layer::Backward.
   * @param bottom input Blob vector (length 1)
   *   -# @f$ (N \times C \times H \times W) @f$
   *      the inputs @f$ x @f$; Backward fills their diff with
   *      gradients @f$
   *        \frac{\partial E}{\partial x} = \left\{
   *        \begin{array}{lr}
   *            0 & \mathrm{if} \; x \le 0 \\
   *            \frac{\partial E}{\partial y} & \mathrm{if} \; x > 0
   *        \end{array} \right.
   *      @f$ if propagate_down[0], by default.
   *      If a non-zero negative_slope @f$ \nu @f$ is provided,
   *      the computed gradients are @f$
   *        \frac{\partial E}{\partial x} = \left\{
   *        \begin{array}{lr}
   *            \nu \frac{\partial E}{\partial y} & \mathrm{if} \; x \le 0 \\
   *            \frac{\partial E}{\partial y} & \mathrm{if} \; x > 0
   *        \end{array} \right.
   *      @f$.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
};

}  // namespace caffe

#endif  // CAFFE_RELUX_LAYER_HPP_
