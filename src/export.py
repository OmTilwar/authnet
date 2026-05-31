"""
AuthNet ONNX Export, INT8 Quantization & Edge Benchmarking
Exports trained model to ONNX → applies INT8 Post-Training Quantization → benchmarks.
"""

import os
import sys
import json
import time
from typing import Optional, Dict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.model import EmbeddingNet, load_model
from src.dataset import get_test_transforms


def export_to_onnx(
    model: EmbeddingNet,
    output_path: Optional[str] = None,
    opset_version: int = 17,
) -> str:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: Trained EmbeddingNet
        output_path: Path for the ONNX file
        opset_version: ONNX opset version
    
    Returns:
        Path to saved ONNX model
    """
    output_path = output_path or config.ONNX_FP32_PATH
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    model.eval()
    model = model.cpu()  # Export on CPU for compatibility
    
    # Dummy input
    dummy_input = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    
    print(f"\nExporting model to ONNX (opset {opset_version})...")
    
    # Force UTF-8 encoding for PyTorch's internal ONNX export prints
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['embedding'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'embedding': {0: 'batch_size'},
        },
        dynamo=False,  # Use legacy TorchScript exporter (stable on Windows)
    )
    
    # Validate ONNX model
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    model_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  [OK] ONNX FP32 model saved: {output_path}")
    print(f"  [OK] Model size: {model_size_mb:.1f} MB")
    
    return output_path


def validate_onnx(
    pytorch_model: EmbeddingNet,
    onnx_path: str,
    atol: float = 1e-5,
) -> bool:
    """
    Validate ONNX model outputs match PyTorch outputs.
    
    Returns:
        True if outputs match within tolerance
    """
    import onnxruntime as ort
    
    pytorch_model.eval()
    pytorch_model = pytorch_model.cpu()
    
    # Generate test input
    test_input = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    
    # PyTorch inference
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).numpy()
    
    # ONNX Runtime inference
    session = ort.InferenceSession(onnx_path)
    onnx_output = session.run(None, {'input': test_input.numpy()})[0]
    
    # Compare
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    match = max_diff < atol
    
    print(f"\n  ONNX Validation:")
    print(f"    Max absolute difference: {max_diff:.2e}")
    print(f"    Tolerance: {atol:.2e}")
    print(f"    Status: {'[PASS]' if match else '[FAIL]'}")
    
    return match


class CalibrationDataReader:
    """
    Provides calibration data for ONNX Runtime static quantization.
    Uses a subset of training images for activation range calibration.
    """
    
    def __init__(self, calibration_dir: str, num_samples: int = 100):
        self.transform = get_test_transforms()
        self.data_list = []
        self.current_idx = 0
        
        # Collect calibration images
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = []
        
        for root, dirs, files in os.walk(calibration_dir):
            for f in files:
                if os.path.splitext(f)[1].lower() in valid_ext:
                    images.append(os.path.join(root, f))
        
        # Sample subset
        if len(images) > num_samples:
            import random
            random.seed(42)
            images = random.sample(images, num_samples)
        
        print(f"  Calibration: using {len(images)} images")
        
        # Preprocess all calibration images
        for img_path in tqdm(images, desc="  Preprocessing calibration data", ncols=100):
            try:
                img = Image.open(img_path).convert("RGB")
                tensor = self.transform(img).unsqueeze(0).numpy()
                self.data_list.append({'input': tensor})
            except Exception:
                continue
    
    def get_next(self):
        if self.current_idx >= len(self.data_list):
            return None
        data = self.data_list[self.current_idx]
        self.current_idx += 1
        return data
    
    def rewind(self):
        self.current_idx = 0


def quantize_int8(
    fp32_onnx_path: str,
    output_path: Optional[str] = None,
    calibration_dir: Optional[str] = None,
    num_calibration_samples: int = 100,
) -> str:
    """
    Apply INT8 Post-Training Quantization to the ONNX model.
    
    Uses ONNX Runtime's static quantization with calibration data
    for accurate activation range estimation.
    
    Args:
        fp32_onnx_path: Path to FP32 ONNX model
        output_path: Path for quantized model
        calibration_dir: Directory with calibration images
        num_calibration_samples: Number of images for calibration
    
    Returns:
        Path to quantized model
    """
    from onnxruntime.quantization import quantize_static, CalibrationMethod, QuantType
    
    output_path = output_path or config.ONNX_INT8_PATH
    calibration_dir = calibration_dir or os.path.join(config.COMBINED_DIR, "train")
    
    print(f"\nINT8 Post-Training Quantization")
    print(f"  FP32 model: {fp32_onnx_path}")
    print(f"  Calibration dir: {calibration_dir}")
    
    # Create calibration data reader
    calibration_reader = CalibrationDataReader(
        calibration_dir, 
        num_samples=num_calibration_samples,
    )
    
    # Quantize
    print("  Quantizing (this may take a minute)...")
    quantize_static(
        model_input=fp32_onnx_path,
        model_output=output_path,
        calibration_data_reader=calibration_reader,
        quant_format=None,  # Default QDQ format
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    
    # Report size
    fp32_size = os.path.getsize(fp32_onnx_path) / (1024 * 1024)
    int8_size = os.path.getsize(output_path) / (1024 * 1024)
    compression = fp32_size / int8_size
    
    print(f"\n  Quantization Results:")
    print(f"    FP32 model size: {fp32_size:.1f} MB")
    print(f"    INT8 model size: {int8_size:.1f} MB")
    print(f"    Compression:     {compression:.1f}x smaller")
    print(f"    Saved to: {output_path}")
    
    return output_path


def benchmark_inference(
    pytorch_model: EmbeddingNet,
    fp32_onnx_path: str,
    int8_onnx_path: Optional[str] = None,
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict:
    """
    Comprehensive inference benchmarking across all runtimes.
    
    Benchmarks:
    1. PyTorch GPU (FP32)
    2. PyTorch CPU (FP32)
    3. ONNX Runtime CPU (FP32)
    4. ONNX Runtime CPU (INT8) — if quantized model available
    
    Returns:
        Dict with benchmark results
    """
    import onnxruntime as ort
    
    print(f"\n{'='*60}")
    print("  Inference Benchmarking")
    print(f"  Runs: {num_runs} (+ {warmup_runs} warmup)")
    print(f"{'='*60}")
    
    dummy_input_np = np.random.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).astype(np.float32)
    dummy_input_torch = torch.from_numpy(dummy_input_np)
    
    results = {}
    
    # ── 1. PyTorch GPU ──
    if torch.cuda.is_available():
        pytorch_model.eval()
        pytorch_model = pytorch_model.cuda()
        input_gpu = dummy_input_torch.cuda()
        
        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                _ = pytorch_model(input_gpu)
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = pytorch_model(input_gpu)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        
        results['pytorch_gpu'] = {
            'mean_ms': np.mean(times),
            'median_ms': np.median(times),
            'p95_ms': np.percentile(times, 95),
            'std_ms': np.std(times),
        }
        print(f"\n  PyTorch GPU:  {np.median(times):.2f} ms (median)")
    
    # ── 2. PyTorch CPU ──
    pytorch_model = pytorch_model.cpu()
    input_cpu = dummy_input_torch.cpu()
    
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = pytorch_model(input_cpu)
    
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = pytorch_model(input_cpu)
        times.append((time.perf_counter() - start) * 1000)
    
    results['pytorch_cpu'] = {
        'mean_ms': np.mean(times),
        'median_ms': np.median(times),
        'p95_ms': np.percentile(times, 95),
        'std_ms': np.std(times),
    }
    print(f"  PyTorch CPU:  {np.median(times):.2f} ms (median)")
    
    # ── 3. ONNX Runtime CPU (FP32) ──
    session_fp32 = ort.InferenceSession(
        fp32_onnx_path,
        providers=['CPUExecutionProvider'],
    )
    
    for _ in range(warmup_runs):
        _ = session_fp32.run(None, {'input': dummy_input_np})
    
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = session_fp32.run(None, {'input': dummy_input_np})
        times.append((time.perf_counter() - start) * 1000)
    
    results['onnx_fp32_cpu'] = {
        'mean_ms': np.mean(times),
        'median_ms': np.median(times),
        'p95_ms': np.percentile(times, 95),
        'std_ms': np.std(times),
    }
    print(f"  ONNX FP32:   {np.median(times):.2f} ms (median)")
    
    # ── 4. ONNX Runtime CPU (INT8) ──
    if int8_onnx_path and os.path.exists(int8_onnx_path):
        session_int8 = ort.InferenceSession(
            int8_onnx_path,
            providers=['CPUExecutionProvider'],
        )
        
        for _ in range(warmup_runs):
            _ = session_int8.run(None, {'input': dummy_input_np})
        
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = session_int8.run(None, {'input': dummy_input_np})
            times.append((time.perf_counter() - start) * 1000)
        
        results['onnx_int8_cpu'] = {
            'mean_ms': np.mean(times),
            'median_ms': np.median(times),
            'p95_ms': np.percentile(times, 95),
            'std_ms': np.std(times),
        }
        print(f"  ONNX INT8:   {np.median(times):.2f} ms (median)")
    
    # ── Model sizes ──
    fp32_size = os.path.getsize(fp32_onnx_path) / (1024 * 1024)
    results['model_sizes'] = {'fp32_mb': fp32_size}
    
    if int8_onnx_path and os.path.exists(int8_onnx_path):
        int8_size = os.path.getsize(int8_onnx_path) / (1024 * 1024)
        results['model_sizes']['int8_mb'] = int8_size
        results['model_sizes']['compression_ratio'] = fp32_size / int8_size
    
    # ── Parameter count ──
    total_params = sum(p.numel() for p in pytorch_model.parameters())
    results['parameters'] = {
        'total': total_params,
        'total_millions': total_params / 1e6,
    }
    
    # ── Summary Table ──
    print(f"\n{'='*60}")
    print("  Benchmark Summary")
    print(f"{'='*60}")
    print(f"  {'Runtime':<20} {'Median (ms)':<15} {'P95 (ms)':<15}")
    print(f"  {'-'*50}")
    
    for runtime, metrics in results.items():
        if runtime in ('model_sizes', 'parameters'):
            continue
        name = runtime.replace('_', ' ').upper()
        print(f"  {name:<20} {metrics['median_ms']:<15.2f} {metrics['p95_ms']:<15.2f}")
    
    print(f"\n  Model size (FP32): {results['model_sizes']['fp32_mb']:.1f} MB")
    if 'int8_mb' in results.get('model_sizes', {}):
        print(f"  Model size (INT8): {results['model_sizes']['int8_mb']:.1f} MB")
        print(f"  Compression:       {results['model_sizes']['compression_ratio']:.1f}x")
    print(f"  Parameters:        {results['parameters']['total_millions']:.1f}M")
    print(f"{'='*60}")
    
    # Save results
    results_path = os.path.join(config.BENCHMARK_DIR, "benchmark_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")
    
    return results


def full_export_pipeline(
    model: Optional[EmbeddingNet] = None,
    model_path: Optional[str] = None,
):
    """
    Run the complete export pipeline:
    1. Export to ONNX (FP32)
    2. Validate ONNX model
    3. Apply INT8 quantization
    4. Benchmark all runtimes
    """
    # Load model
    if model is None:
        model_path = model_path or config.BEST_MODEL_PATH
        model = load_model(model_path)
    
    print("=" * 60)
    print("  AuthNet Export & Quantization Pipeline")
    print("=" * 60)
    
    # Step 1: Export to ONNX
    fp32_path = export_to_onnx(model)
    
    # Step 2: Validate
    validate_onnx(model, fp32_path)
    
    # Step 3: INT8 Quantization
    int8_path = None
    try:
        int8_path = quantize_int8(fp32_path)
    except Exception as e:
        print(f"\n  [WARN] INT8 quantization failed: {e}")
        print("  Continuing with FP32 benchmark only.")
    
    # Step 4: Benchmark
    model_for_bench = model.to(config.DEVICE)
    results = benchmark_inference(
        pytorch_model=model_for_bench,
        fp32_onnx_path=fp32_path,
        int8_onnx_path=int8_path,
    )
    
    return results


if __name__ == "__main__":
    full_export_pipeline()
