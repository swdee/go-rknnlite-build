# YOLO26 Notes

## Model Source

The YOLO26 model is based on the [Ultralytics YOLO26 fork](https://github.com/zycer/yolo26_rknn_ultralytics) as
the upstream vendor does not support exporting of a RKNN compatible ONNX model due to
Rockchips NPU/Toolkit not supporting the `TopK` operator.

The quantization dataset used are a random selection of images from the [COCO val2017 dataset](https://cocodataset.org/#download).

## Build

From Python 3.12.11

Check out the forked code to your development environment and install into a python virtual environment.
```
git clone https://github.com/zycer/yolo26_rknn_ultralytics.git
cd yolo26_rknn_ultralytics
pip install -e .
pip install onnxscript
```

Export model to an RKNN compatible ONNX format.
```
yolo export model=yolo26s.pt format=rknn opset=19
```

