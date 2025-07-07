import torch
from torch import nn

# 1. import or define your OSNet architecture
#    If you’re using the torchreid package:
#    pip install torchreid
from torchreid import models

def load_osnet(model_path: str, num_classes: int = 1000):
    # build the same OSNet variant you trained (e.g. osnet_x1_0)
    model = models.build_model(
        name='osnet_x1_0',
        num_classes=num_classes,
        pretrained=False
    )
    checkpoint = torch.load(model_path, map_location='cpu')
    # if you saved state_dict only:
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    # strip any 'module.' prefixes if needed
    new_state = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state[new_key] = v
    model.load_state_dict(new_state)
    return model

def export_to_onnx(model: nn.Module, onnx_path: str,
                   input_size=(1, 3, 256, 128),
                   dynamic_batch=True,
                   opset=11):
    model.eval()
    dummy_input = torch.randn(*input_size)
    # set dynamic axes for batch dimension if desired
    dyn_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} if dynamic_batch else None

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        #dynamic_axes=dyn_axes
    )
    print(f"ONNX model saved to {onnx_path}")

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('pth',   help='Path to your OSNet .pth checkpoint')
    p.add_argument('onnx',  help='Where to save the ONNX model')
    p.add_argument('--h',   type=int, default=256, help='Input height')
    p.add_argument('--w',   type=int, default=128, help='Input width')
    p.add_argument('--no-dyn', action='store_false', dest='dyn', help='Disable dynamic batch size')
    p.add_argument('--opset', type=int, default=13, help='ONNX opset version')
    p.add_argument('--c', type=int, default=751, help='Number of classes')
    args = p.parse_args()

    osnet = load_osnet(args.pth, args.c)
    export_to_onnx(
        osnet,
        args.onnx,
        input_size=(1, 3, args.h, args.w),
        dynamic_batch=args.dyn,
        opset=args.opset
    )
