import os
import sys
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import minimize

from torchvision.transforms import Compose

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_depth_image(depth, path, grayscale=False):
    norm_depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    norm_depth = norm_depth.astype(np.uint8)

    if grayscale:
        norm_depth = np.repeat(norm_depth[..., np.newaxis], 3, axis=-1)
    else:
        norm_depth = cv2.applyColorMap(norm_depth, cv2.COLORMAP_INFERNO)

    cv2.imwrite(path, norm_depth)

class DepthEstimator:
    def __init__(self, dataset, depth_dir, model_type="depth_anything_v2", load_model=True):
        # Load depth estimates if they already exist
        stored_depths = dict()
        raw_dir = str(Path(depth_dir) / "raw")
        opt_dir = str(Path(depth_dir) / "optimized")
        if not os.path.exists(depth_dir):
            os.makedirs(depth_dir)
            os.makedirs(str(Path(depth_dir) / "images" / "dense"))
            os.makedirs(str(Path(depth_dir) / "images" / "sparse"))
            os.makedirs(str(Path(depth_dir) / "images" / "optimized"))
            os.makedirs(str(Path(depth_dir) / "raw"))
            os.makedirs(str(Path(depth_dir) / "optimized"))
        elif os.path.exists(depth_dir):
            for file_name in tqdm(os.listdir(opt_dir)):
                if file_name.endswith('.npy'):
                    stored_depths[file_name[:-4]] = np.load(os.path.join(depth_dir, file_name), allow_pickle=True)
        self.depth_dir = depth_dir
        self.imgs_dir = Path(self.depth_dir) / "images"

        # Load the model if not all images have been processed
        self.model_type = model_type
        if len(stored_depths) < len(dataset) or load_model:
            self.load_model(model_type)

        # Check depth for every camera
        for idx in tqdm(range(len(dataset))):
            camera_name = Path(dataset.parser.image_names[idx]).stem
            depth = stored_depths.get(camera_name)
            if depth is not None:
                continue
            else:
                raw_depth, optimized_depth = self.estimate(idx, camera_name, dataset)
                np.save(os.path.join(raw_dir, camera_name + '.npy'), raw_depth)
                np.save(os.path.join(opt_dir, camera_name + '.npy'), optimized_depth)

    def load_model(self, model_type="depth_anything_v2"):
        if model_type == "depth_anything_v2":
            self.depth_model = DepthAnythingV2Model(device)
        else:
            raise ValueError("Unknown depth model type")

    def estimate(self, idx, camera_name, dataset):
        "Returns a SfM scale-matched dense depth map for a chosen camera"

        # Estimate sparse and dense depth (and reprojection error) maps
        D_sparse, E_sparse = self._estimate_sparse(idx, dataset)
        D_dense = self._estimate_dense(idx, dataset).squeeze()

        # save images
        save_depth_image(D_sparse.toarray(), str(self.imgs_dir / "sparse" / f"{camera_name}.png"))
        save_depth_image(D_dense, str(self.imgs_dir / "dense" / f"{camera_name}.png"))

        # match scale
        D_optimized = self._match_scale(D_dense, D_sparse, E_sparse)

        # save final depth
        save_depth_image(D_optimized.squeeze(), str(self.imgs_dir / "optimized" / f"{camera_name}.png"))

        return D_dense, D_optimized

    def _estimate_dense(self, idx, dataset):
        "Use a monocular depth estimation model to estimate depth"
        if self.depth_model is None:
            raise ValueError("No depth model loaded")
        return self.depth_model.predict(dataset[idx]["image_fp"])

    def _estimate_sparse(self, idx, dataset):
        data = dataset[idx]
        image_height, image_width = data["image"].size()[0], data["image"].size()[1]

        # obtain sfm points projected to image plane and their depths
        pts = data["points"].int()
        d = data["depths"]
        errs = data["points_errs"]

        x_2d = pts[:, 0]
        y_2d = pts[:, 1]
        z = d

        # Create sparse depth and error maps
        D_sparse = lil_matrix((image_height, image_width))
        E_sparse = lil_matrix((image_height, image_width))

        for x_, y_, z_, e_ in zip(x_2d, y_2d, z, errs):
            if 0 <= x_ < image_width and 0 <= y_ < image_height:
                D_sparse[y_, x_] = z_
                E_sparse[y_, x_] = e_
        D_sparse = D_sparse.tocoo()
        E_sparse = E_sparse.tocoo()

        return D_sparse, E_sparse

    def _match_scale(self, D_dense, D_sparse, E_sparse=None):
        "Matches the scale in the provided metric depth map to that in the sparse map"
        i, j = D_sparse.row, D_sparse.col
        z_dense = torch.as_tensor(D_dense[i,j].data).to(device)
        z_sparse = torch.as_tensor(D_sparse.data).to(device)

        def func(args):
            s, t = args[0], args[1]
            z_dense_adj = s * z_dense + t
            return (z_sparse - z_dense_adj).abs().mean().cpu().numpy()
        res = minimize(func, x0=[-0.5,3], method='Nelder-Mead')

        s, t = res.x
        D_dense = s * D_dense + t
        return D_dense


class DepthAnythingV2Model:
    def __init__(self, device):
        sys.path.append(os.path.expanduser("~/code/mast3r/Depth-Anything-V2"))

        from depth_anything_v2.dpt import DepthAnythingV2

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

        self.model = DepthAnythingV2(**model_configs[encoder])
        self.model.load_state_dict(torch.load(f'Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
        self.model.to(device).eval()

    def predict(self, img_fp):
        raw_img = cv2.imread(img_fp)
        h, w, c = raw_img.shape
        with torch.no_grad():
            depth = self.model.infer_image(raw_img, input_size=h)

        return depth
