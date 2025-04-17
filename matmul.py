import argparse
import time

import numpy as np
import torch
from torch import mps, cuda


def synchronize(device):
    if device.type == "cuda":
        cuda.synchronize()
    elif device.type == "mps":
        mps.synchronize()
    elif device.type == "cpu":
        pass


def matmul(arr, num, device):
    start = time.time()
    synchronize(device)
    for i in range(num):
        torch.matmul(arr, arr)
    synchronize(device)
    return time.time() - start


def flops_benchmark(device, dtype, num_try=10000):
    test_range = 2 ** np.arange(10, 14, 1)
    results = []
    for n in test_range:
        n = int(n)
        a = torch.rand(n, n, device=device, dtype=dtype)
        matmul(a, 5, device)
        total_time = matmul(a, num_try, device)
        tflops = 2 * n**3 / total_time / 1e12 * num_try
        results.append(tflops)
    return results


def mem_copy(a, b, num_try, device):
    start = time.time()
    synchronize(device)
    for _ in range(num_try):
        a.copy_(b)
    synchronize(device)
    return time.time() - start


def memory_bandwidth_benchmark(device, num_try=100):  # 256MB
    test_range = 2 ** (np.arange(20, 30, 1))
    best_bandwidth = 0
    for size in test_range:
        elapsed_time = 0
        # Create random tensors
        a = torch.rand(size, device=device)
        b = torch.rand(size, device=device)
        mem_copy(a, b, 5, device)
        elapsed_time = mem_copy(a, b, num_try, device)
        # Calculate Bandwidth in GB/s
        bytes_copied = a.nelement() * a.element_size()  # bytes
        bandwidth = 2 * bytes_copied / elapsed_time / 1e9 * num_try  # GB/s
        best_bandwidth = max(best_bandwidth, bandwidth)
    return best_bandwidth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "-d", type=str, default="cuda:0")
    parser.add_argument("--num", "-n", type=int, default=5000)
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Benchmarking TFLOPs with FP32")
    fp32_flops = flops_benchmark(device, torch.float32, args.num)
    print("Benchmarking TFLOPs with FP16")
    fp16_flops = flops_benchmark(device, torch.float16, args.num * 2)
    print("Benchmarking Bandwidth")
    bandwidth = memory_bandwidth_benchmark(device)

    results = fp32_flops + fp16_flops + [bandwidth]
    print("\t".join([f"{o:.4f}" for o in results]))


if __name__ == '__main__':
    main()

