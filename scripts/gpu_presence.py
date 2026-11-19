"""Hold a small CUDA tensor briefly and print a timestamped allocation record."""

import argparse
from datetime import datetime, timezone
import os
import socket
import time

import torch


def log(message):
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"{timestamp} {message}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA-visible GPU index (default: 0)")
    parser.add_argument("--seconds", type=int, default=5, help="hold time in seconds (default: 5)")
    args = parser.parse_args()
    if args.seconds < 1:
        parser.error("--seconds must be positive")
    if not 0 <= args.gpu < torch.cuda.device_count():
        parser.error(f"cuda:{args.gpu} is unavailable")

    device = torch.device("cuda", args.gpu)
    buffer = torch.zeros(16 * 1024**2, dtype=torch.uint8, device=device)
    try:
        torch.cuda.synchronize(device)
        log(f"allocated host={socket.gethostname()} pid={os.getpid()} device={device} "
            f"gpu=\"{torch.cuda.get_device_name(device)}\" tensor_bytes={buffer.nbytes} hold_seconds={args.seconds}")
        time.sleep(args.seconds)
    finally:
        del buffer
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
        log(f"released pid={os.getpid()} device={device} tensor_bytes_remaining={torch.cuda.memory_allocated(device)}")


if __name__ == "__main__":
    main()
