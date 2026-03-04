"""
evaluation/measure_end2end_latency.py — End-to-end pipeline latency measurement.

Measures per-sample latency for: backbone forward → quality → geo distance → fusion.
Performs warm-up runs before timing, and GPU synchronization when applicable.

Usage:
    python evaluation/measure_end2end_latency.py --out results/latency.json
    python evaluation/measure_end2end_latency.py --model-path results/face/adaptive/best_model.pt
"""

import argparse
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger('latency')

WARMUP_RUNS = 50
TIMING_RUNS = 200


def collect_hw_info() -> Dict[str, Any]:
    hw: Dict[str, Any] = {
        'hostname': platform.node(),
        'platform': platform.platform(),
        'cpu': platform.uname().processor or platform.uname().machine,
        'python': sys.version,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    try:
        import torch
        hw['torch_version'] = torch.__version__
        hw['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            hw['cuda_version'] = torch.version.cuda
            hw['gpu_names'] = [torch.cuda.get_device_name(i)
                               for i in range(torch.cuda.device_count())]
    except ImportError:
        hw['torch'] = 'not installed'
    try:
        import psutil
        hw['ram_total_gb'] = round(psutil.virtual_memory().total / 1e9, 2)
    except ImportError:
        pass
    try:
        import subprocess
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            hw['nvidia_smi'] = r.stdout.strip()
    except Exception:
        pass
    return hw


def make_fake_input(batch_size: int = 1, image_size: int = 112):
    """Generate a fake face image tensor."""
    import torch
    return torch.randn(batch_size, 3, image_size, image_size)


def make_fake_geo_input():
    """Generate fake geolocation features."""
    import torch
    # [vehicle_lat, vehicle_lon, user_lat, user_lon, distance_km, time_sin, time_cos]
    return torch.randn(1, 7)


def run_latency_benchmark(
    model_path: str = None,
    out_path: str = 'results/latency.json',
    warmup: int = WARMUP_RUNS,
    timing: int = TIMING_RUNS,
) -> Dict[str, Any]:
    import torch
    import torch.nn.functional as F

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    logger.info(f"Warm-up runs: {warmup}, Timing runs: {timing}")

    # Load or create face backbone
    try:
        from src.models.embedding_extractor import EmbeddingExtractor
        extractor = EmbeddingExtractor(
            backbone_name='resnet50', embedding_dim=512, device=device
        )
        extractor.eval()
        logger.info("Loaded EmbeddingExtractor (ResNet-50 backbone)")
        backbone_available = True
    except Exception as e:
        logger.warning(f"Could not load EmbeddingExtractor: {e}. Using random embeddings.")
        backbone_available = False

    # Optionally load quality estimator
    try:
        from src.utils.quality_metrics import compute_quality_score
        quality_fn_available = True
        logger.info("Quality estimator loaded")
    except Exception:
        quality_fn_available = False
        logger.warning("Quality estimator not available; using constant quality=0.8")

    def run_pipeline(face_input: 'torch.Tensor') -> float:
        """
        Run one full pipeline step:
          1. Backbone forward → embedding
          2. Quality estimate
          3. Geo distance (simulated)
          4. Fusion score

        Returns fusion score (float).
        """
        # Step 1: Backbone forward
        if backbone_available:
            with torch.no_grad():
                embedding = extractor.extract(face_input.to(device))
        else:
            embedding = F.normalize(torch.randn(1, 512, device=device), dim=1)

        # Step 2: Quality (image quality metric)
        if quality_fn_available:
            # Quality runs on CPU numpy; convert 112x112 input
            img_np = (face_input[0].permute(1, 2, 0).cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
            quality = compute_quality_score(img_np)
        else:
            quality = 0.8

        # Step 3: Geo distance (haversine, simulated)
        # vehicle @ (40.748, -73.967), user @ slightly offset
        from math import radians, sin, cos, sqrt, atan2
        lat1, lon1 = 40.748, -73.967
        lat2, lon2 = 40.749 + np.random.normal(0, 0.001), -73.966 + np.random.normal(0, 0.001)
        R = 6371.0
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        dist_km = 2 * R * atan2(sqrt(a), sqrt(1 - a))
        sigma = 0.5  # km
        T_geo = float(np.exp(-0.5 * (dist_km / sigma) ** 2))

        # T_face from embedding cosine similarity to a random reference
        ref_emb = F.normalize(torch.randn(1, 512, device=device), dim=1)
        cos_sim = float(F.cosine_similarity(embedding, ref_emb).item())
        T_face = float(1.0 / (1.0 + np.exp(-(cos_sim - 0.3) / 0.1)))

        # Step 4: Fusion
        T_fusion = 0.6 * T_face + 0.4 * T_geo

        # GPU sync before stopping timer (called by caller)
        return T_fusion

    fake_input = make_fake_input()

    # --- Warm-up ---
    logger.info(f"Running {warmup} warm-up iterations...")
    for _ in range(warmup):
        run_pipeline(fake_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # --- Timing ---
    logger.info(f"Running {timing} timed iterations...")
    latencies_ms = []

    for _ in range(timing):
        face_input = make_fake_input()  # fresh input each time

        t0 = time.perf_counter()
        _ = run_pipeline(face_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies_ms.append((t1 - t0) * 1000.0)

    latencies = np.array(latencies_ms)

    results = {
        'hardware': collect_hw_info(),
        'config': {
            'warmup_runs': warmup,
            'timing_runs': timing,
            'backbone': 'resnet50' if backbone_available else 'random_512d',
            'device': str(device),
            'pipeline_steps': ['backbone_forward', 'quality_estimation', 'geo_distance', 'score_fusion'],
        },
        'latency_ms': {
            'mean':   float(np.mean(latencies)),
            'std':    float(np.std(latencies)),
            'min':    float(np.min(latencies)),
            'max':    float(np.max(latencies)),
            'p50':    float(np.percentile(latencies, 50)),
            'p95':    float(np.percentile(latencies, 95)),
            'p99':    float(np.percentile(latencies, 99)),
        },
        'throughput_fps': float(1000.0 / np.mean(latencies)),
    }

    logger.info("\nLatency Results:")
    lm = results['latency_ms']
    logger.info(f"  Mean:  {lm['mean']:.3f} ms")
    logger.info(f"  Std:   {lm['std']:.3f} ms")
    logger.info(f"  Min:   {lm['min']:.3f} ms")
    logger.info(f"  Max:   {lm['max']:.3f} ms")
    logger.info(f"  p50:   {lm['p50']:.3f} ms")
    logger.info(f"  p95:   {lm['p95']:.3f} ms")
    logger.info(f"  p99:   {lm['p99']:.3f} ms")
    logger.info(f"  Throughput: {results['throughput_fps']:.1f} fps")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved latency results → {out}")

    return results


def main():
    parser = argparse.ArgumentParser(description="End-to-end pipeline latency measurement")
    parser.add_argument('--model-path', default=None,
                        help='Optional path to face backbone checkpoint')
    parser.add_argument('--out', default='results/latency.json',
                        help='Output JSON path')
    parser.add_argument('--warmup', type=int, default=WARMUP_RUNS)
    parser.add_argument('--timing', type=int, default=TIMING_RUNS)
    args = parser.parse_args()

    run_latency_benchmark(
        model_path=args.model_path,
        out_path=args.out,
        warmup=args.warmup,
        timing=args.timing,
    )


if __name__ == '__main__':
    main()
