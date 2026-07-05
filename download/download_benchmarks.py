import argparse
from huggingface_hub import snapshot_download
import os

BENCHMARKS = [
    "uoft-cs/cifar10",
]

aliases = {
    "cifar10": "uoft-cs/cifar10",
}

benchmark_map = {m.split("/")[-1]: m for m in BENCHMARKS}
benchmark_map.update(aliases)

parser = argparse.ArgumentParser(description="Download datasets")
parser.add_argument(
    "-a", "--all",
    action="store_true",
    help="Download all benchmarks",
)
parser.add_argument(
    "-b", "--benchmarks",
    nargs="+",
    metavar="BENCHMARK",
    help=f"Benchmarks to download (short names: {', '.join(aliases)}).",
)
parser.add_argument(
    "-d", "--dir",
    default="./data",
    help="Download directory (default: ./data)",
)
args = parser.parse_args()

if not args.all and not args.benchmarks:
    parser.print_help()
    exit(1)

os.makedirs(args.dir, exist_ok=True)

if args.all:
    selected = list(BENCHMARKS)
else:
    selected = []
    for name in args.benchmarks:
        if name in benchmark_map:
            selected.append(benchmark_map[name])
        else:
            print(f"Unknown benchmark: {name}, skipping.")

for bench_id in selected:
    print(f"Downloading {bench_id} ...")

    data_dir = snapshot_download(
        repo_id=bench_id,
        repo_type="dataset",
        local_dir=f"{args.dir}/{bench_id.split('/')[-1]}",
        resume_download=True,
    )
    print(f"Downloaded to {data_dir}")

print("All done.")