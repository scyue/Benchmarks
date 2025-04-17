#!/usr/bin/env python3
"""
Distributed ResNet50 synthetic benchmark to measure multi-GPU throughput with minimal CPU overhead.

Usage:
  # via torchrun (recommended):
  torchrun --nproc_per_node=NUM_GPUS distributed_resnet50_synth.py \
      --per-gpu-batch-size 2048 --duration 180 --report-interval 30

  # direct python (spawns one process per GPU):
  python distributed_resnet50_synth.py --per-gpu-batch-size 2048 --duration 180 --report-interval 30

This script avoids any DataLoader overhead by reusing a single synthetic batch each iteration.
It automatically scales the total batch size based on per-GPU batch size and number of GPUs.
It runs for a fixed duration (default: 180s) and reports throughput every report-interval seconds.
"""
import argparse
import time
import os
import logging
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models, ops as vision_ops

# Logging setup for rank 0 only
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s][Rank %(rank)d][%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# Monkey-patch missing NMS operator if necessary
if not hasattr(vision_ops, 'nms'):
    from torch import Tensor
    @torch.jit.unused
    def _stub_nms(boxes: Tensor, scores: Tensor, iou_threshold: float):
        return torch.arange(boxes.size(0), device=boxes.device)
    vision_ops.nms = _stub_nms


def parse_args():
    parser = argparse.ArgumentParser(description='Distributed ResNet50 synthetic benchmark')
    parser.add_argument('--per-gpu-batch-size', type=int, default=2048,
                        help='batch size per GPU (default: 2048)')
    parser.add_argument('--duration', type=int, default=180,
                        help='benchmark duration in seconds (default: 180)')
    parser.add_argument('--report-interval', type=int, default=30,
                        help='seconds between throughput reports (default: 30)')
    parser.add_argument('--backend', type=str, default='nccl',
                        help='distributed backend')
    parser.add_argument('--init-method', type=str, default='env://',
                        help='init method for process group')
    parser.add_argument('--world-size', type=int, default=None,
                        help='override total number of processes (GPUs); usually not needed')
    return parser.parse_args()


def main_worker(local_rank, world_size, args):
    # Prepare logging for rank 0
    log = logging.LoggerAdapter(logger, {'rank': local_rank})
    if local_rank != 0:
        log.logger.disabled = True

    # Init distributed group
    dist.init_process_group(backend=args.backend,
                            init_method=args.init_method,
                            world_size=world_size,
                            rank=local_rank)

    # Device setup
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}; CUDA available: {torch.cuda.is_available()}")

    # Batch sizes
    per_gpu_bs = args.per_gpu_batch_size
    total_bs = per_gpu_bs * world_size
    log.info(f"Per-GPU batch size: {per_gpu_bs}, Total batch size: {total_bs}")
    log.info(f"Duration: {args.duration}s, Report interval: {args.report_interval}s")

    # Synthetic data
    synth_images = torch.randn((per_gpu_bs, 3, 32, 32), device=device)
    synth_labels = torch.zeros((per_gpu_bs,), dtype=torch.long, device=device)

    # Model and optimizer
    model = models.resnet50().to(device)
    model = DDP(model, device_ids=[local_rank])
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                                momentum=0.9, weight_decay=1e-4)

    # Benchmark
    start_time = time.time()
    last_report = start_time
    images_since_last = 0
    total_images = 0

    log.info("Starting synthetic benchmark loop.")
    while True:
        now = time.time()
        elapsed = now - start_time
        if elapsed >= args.duration:
            break

        # Forward-backward step
        outputs = model(synth_images)
        loss = criterion(outputs, synth_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        images_since_last += per_gpu_bs
        total_images += per_gpu_bs

        # Reporting
        if local_rank == 0 and now - last_report >= args.report_interval:
            interval = now - last_report
            throughput = images_since_last * world_size / interval
            log.info(f"[Time {elapsed:.1f}s] {throughput:.2f} imgs/sec over last {interval:.1f}s")
            last_report = now
            images_since_last = 0

    # Final summary
    dist.barrier()
    total_time = time.time() - start_time
    agg_throughput = total_images * world_size / total_time
    log.info("==============================")
    log.info(f"Total time: {total_time:.2f}s, Aggregate throughput: {agg_throughput:.2f} imgs/sec")
    log.info(f"Total images processed: {total_images * world_size}")
    log.info("==============================")

    dist.destroy_process_group()
    log.info("Benchmark complete, exiting.")


def main():
    args = parse_args()
    # Detect torchrun
    local_rank_env = os.environ.get('LOCAL_RANK')
    world_size_env = os.environ.get('WORLD_SIZE')
    if local_rank_env is not None and world_size_env is not None:
        local_rank = int(local_rank_env)
        world_size = int(world_size_env) if args.world_size is None else args.world_size
        main_worker(local_rank, world_size, args)
    else:
        world_size = args.world_size or torch.cuda.device_count()
        mp.spawn(main_worker, nprocs=world_size, args=(world_size, args,))

if __name__ == '__main__':
    main()

