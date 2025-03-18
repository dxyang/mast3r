import os
import logging
from argparse import ArgumentParser
import shutil
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms

'''
python center_crop.py \
--source_path /srv/warplab/dxy/gopro_slam/2024_11_USVI/2024_11_14_yawzi/metashape_export/images \
--dest_path /srv/warplab/dxy/gopro_slam/2024_11_USVI/2024_11_14_yawzi/metashape_export/images_cc
'''

def process_images(source_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    files = os.listdir(source_dir)

    saved_width, saved_height = None, None

    for idx, file in enumerate(tqdm(files)):
        source_file = os.path.join(source_dir, file)
        dest_file = os.path.join(dest_dir, file)

        image = Image.open(source_file)
        if image is None:
            print(f"Failed to read {source_file}")
            continue

        width, height = image.size
        if idx == 0:
            saved_width, saved_height = width, height
        assert saved_width == width and saved_height == height
        transform = transforms.CenterCrop((height // 2, width // 2))
        cropped_image = transform(image)
        cropped_image.save(dest_file)

if __name__ == "__main__":
    parser = ArgumentParser("Center cropper")
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--dest_path", "-d", required=True, type=str)
    args = parser.parse_args()

    process_images(args.source_path, args.dest_path)