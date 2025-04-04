
from pathlib import Path
from tap import Tap
import torch

from datasets.colmap import Dataset, Parser

from .depth import DepthEstimator

class DepthArgParser(Tap):
    data_dir: str
    factor: int = 1
    center_crop: bool = False

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = DepthArgParser().parse_args()

    parser = Parser(
        args.data_dir,
        factor=args.factor,
        normalize=False,
        test_every=1e10,
        center_crop=args.center_crop,
    )
    dataset = Dataset(
        parser,
        split="all",
        load_depths=True,
    )

    if args.factor == 1:
        depth_path = Path(args.data_dir) / "depth"
    elif args.factor == 2:
        depth_path = Path(args.data_dir) / "depth_2"
        if args.center_crop:
            depth_path = Path(args.data_dir) / "depth_cc"

    depth_estimator = DepthEstimator(dataset, depth_path)
