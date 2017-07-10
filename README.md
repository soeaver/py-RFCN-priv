# py-RFCN-priv
py-RFCN-priv is based on [py-R-FCN-multiGPU](https://github.com/bharatsingh430/py-R-FCN-multiGPU), thanks for bharatsingh430's job.


### Disclaimer

The official R-FCN code (written in MATLAB) is available [here](https://github.com/daijifeng001/R-FCN).

py-R-FCN is modified from [the offcial R-FCN implementation](https://github.com/daijifeng001/R-FCN) and  [py-faster-rcnn code](https://github.com/rbgirshick/py-faster-rcnn ), and the usage is quite similar to [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn ).

py-R-FCN-multiGPU is a modified version of [py-R-FCN](https://github.com/Orpine/py-R-FCN), the original code is available [here](https://github.com/bharatsingh430/py-R-FCN-multiGPU).

py-RFCN-priv also supports [soft-nms](https://github.com/bharatsingh430/soft-nms).

caffe-priv supports [depthwise convolution](https://github.com/yonghenglh6/DepthwiseConvolution), [roi warping](https://github.com/daijifeng001/caffe-mnc), [roi mask pooling](https://github.com/craftGBD/caffe-GBD), [bilinear interpolation](https://bitbucket.org/deeplab/deeplab-public/).


### New features

py-RFCN-priv supports:
 - Label shuffling (only single GPU training);
 - PIXEL_STD;
 - Anchors outside image (described in [FPN](https://arxiv.org/abs/1612.03144));
 - Performing bilinear interpolation operator accoording to input blobs size.
 
 
### Installation

1. Clone the py-RFCN-priv repository
  ```Shell
  git clone https://github.com/soeaver/py-RFCN-priv
  ```
  We'll call the directory that you cloned py-RFCN-priv into `RFCN_ROOT`

2. Build the Cython modules
    ```Shell
    cd $RFCN_ROOT/lib
    make
    ```
3. Build Caffe and pycaffe
    ```Shell
    cd $RFCN_ROOT/caffe-priv
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html
    
    # cp Makefile.config.example Makefile.config
    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make all -j && make pycaffe -j
   ```    
   
   **Note:** Caffe *must* be built with support for Python layers!
   ```make
   # In your Makefile.config, make sure to have this line uncommented
   WITH_PYTHON_LAYER := 1
   # Unrelatedly, it's also recommended that you use CUDNN
   USE_CUDNN := 1
   USE_NCCL := 1

### License

py-RFCN-priv and caffe-priv are released under the MIT License (refer to the LICENSE file for details).
