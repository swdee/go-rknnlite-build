#!/usr/bin/env python3
import argparse
import inspect
import sys
from pathlib import Path

import torch


class MidasWrap(torch.nn.Module):
    def __init__(self, midas_model: torch.nn.Module):
        super().__init__()
        self.m = midas_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.m(x)
        # Many MiDaS models return (N,H,W); make it (N,1,H,W) for a clean output
        if y.dim() == 3:
            y = y.unsqueeze(1)
        return y


def torch_export_onnx_single_file(model, dummy, onnx_path: Path, opset: int):
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # Build kwargs in a version-safe way (torch signatures vary)
    sig = inspect.signature(torch.onnx.export)
    kwargs = dict(
        f=str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["depth"],
        dynamic_axes=None,
    )

    # Try to force "no external data" for torch versions that support it
    if "use_external_data_format" in sig.parameters:
        kwargs["use_external_data_format"] = False

    # Newer torch has large_model flag; keep it False
    if "large_model" in sig.parameters:
        kwargs["large_model"] = False

    torch.onnx.export(model, dummy, **kwargs)


def repack_if_external_data(onnx_path: Path, delete_sidecar: bool):
    # Torch usually writes: <name>.onnx and <name>.onnx.data
    data_path = onnx_path.with_suffix(onnx_path.suffix + ".data")
    if not data_path.exists():
        return

    # Merge external weights back into a single ONNX file
    try:
        import onnx
    except Exception as e:
        raise RuntimeError(
            "ONNX external data file was created, but 'onnx' package is not installed.\n"
            "Install it in this export environment with: python -m pip install onnx\n"
            f"Original import error: {e}"
        )

    print(f"Found external weights: {data_path.name}")
    print("Repacking to a single ONNX file...")

    m = onnx.load_model(str(onnx_path), load_external_data=True)
    # Save back to the SAME path, but with embedded tensors
    onnx.save_model(m, str(onnx_path), save_as_external_data=False)

    if delete_sidecar:
        try:
            data_path.unlink()
            print(f"Deleted sidecar: {data_path.name}")
        except Exception as e:
            print(f"WARNING: couldn't delete {data_path.name}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midas-repo", default="midas", help="Path to local isl-org/MiDaS repo root")
    ap.add_argument("--pt", required=True, help="Path to dpt_swin2_tiny_256.pt")
    ap.add_argument("--onnx", default="dpt_swin2_tiny_256.onnx", help="Output ONNX file path (single file)")
    ap.add_argument("--opset", type=int, default=19, help="ONNX opset (12-19 typically OK for RKNN)")
    ap.add_argument("--keep-data", action="store_true",
                    help="Do NOT delete .onnx.data if torch creates it (default deletes after repack)")
    args = ap.parse_args()

    midas_repo = Path(args.midas_repo).resolve()
    pt_path = Path(args.pt).resolve()
    onnx_path = Path(args.onnx).resolve()

    if not midas_repo.exists():
        raise FileNotFoundError(midas_repo)
    if not pt_path.exists():
        raise FileNotFoundError(pt_path)

    # Import MiDaS loader from your local repo
    sys.path.insert(0, str(midas_repo))
    from midas.model_loader import load_model

    device = torch.device("cpu")
    model_type = "dpt_swin2_tiny_256"

    model, _transform, net_w, net_h = load_model(
        device=device,
        model_path=str(pt_path),
        model_type=model_type,
        optimize=False,
        height=None,
        square=False,
    )
    model.eval()

    wrapped = MidasWrap(model).eval()
    dummy = torch.randn(1, 3, net_h, net_w, dtype=torch.float32)

    print(f"Exporting ONNX -> {onnx_path}")
    print(f"Input size: {net_w}x{net_h}")
    torch_export_onnx_single_file(wrapped, dummy, onnx_path, opset=args.opset)

    repack_if_external_data(onnx_path, delete_sidecar=(not args.keep_data))

    print("Done.")
    print("Output:", onnx_path)


if __name__ == "__main__":
    main()
