
# OSNet Notes

## Model Source

The OSNet model was download from
https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html

Download by clicking the `market1501` link for the `osnet_x1_0` under the `Same-domain ReID` table.

The Market1501 training images that form the dataset for quantization comes from
https://github.com/sybernix/market1501


## Build

The osnet_to_onnx.py script has the following python virtual environment requirements;

From Python 3.10.15

Python Dependencies
```
pip install torchvision Pillow torchreid tensorboard onnx gdown torch scipy opencv-python
```

To export .pth to onnx
```
python osnet_to_onnx.py osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth \
  osnet_x1_0_market_256x128.onnx
```