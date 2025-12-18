
# MiDaS Depth Estimation Notes

## Model Source

The MiDaS model used is the v3.1 Swin2-Tiny variation `dpt_swin2_tiny_256` 
designed for embedded applications and is available for download
from the [official project](https://github.com/isl-org/MiDaS) 
homepage.

The quantization dataset used are a random selection of images
from the [ImageNet dataset](https://www.image-net.org/).

The MiDaS source python used for loading pytorch model is cloned from 
https://github.com/isl-org/MiDaS into subdirectory midas


## Build

The midas_to_onnx.py script has the following python virtual environment requirements:

From Python 3.10.19

Python Dependencies
```
pip install timm
```


To export the Pytorch model to Onnx
```
wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_tiny_256.pt
python midas_to_onnx.py --pt dpt_swin2_tiny_256.pt --onnx dpt_swin2_tiny_256.onnx
```