import argparse
import os

import numpy as np
import torch
from unhcv.common import visual_mask, write_im, remove_dir
from unhcv.common.image import concat_differ_size
from unhcv.common.utils import obj_load, ProgressBarTqdm, find_path, attach_home_root
from unhcv.projects.diffusion.inpainting.evaluation.dataset import InpaintEvalutionDataset, InpaintEvalutionDatasetRead
from unhcv.projects.diffusion.inpainting.evaluation.evaluation_model import init_inpainting_eval_dataset

from app import PowerPaintController

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--show_root', type=str, default='show/compare/powerpaint')
    parser.add_argument(
        '--data_indexes_path', type=str, default=None)
    parser.add_argument(
        '--show', action="store_true")
    args = parser.parse_args()
    return args


class CustomPowerPaint:
    def __init__(self, checkpoint_dir="model/PowerPaint-v2-1", version="ppt-v2"):
        weight_dtype = "float16"
        # initialize the pipeline controller
        weight_dtype = torch.float16 if weight_dtype == "float16" else torch.float32
        checkpoint_dir = find_path(checkpoint_dir)
        local_files_only = True
        controller = PowerPaintController(weight_dtype, checkpoint_dir, local_files_only, version)
        self.controller = controller

    def __call__(self, image, mask):
        image = image.convert("RGB")
        mask = mask.convert("RGB")
        input_image = dict(image=image, mask=mask)
        out = self.controller.infer_removal(input_image)
        return out[0][0], out[1][0]

if __name__ == "__main__":
    args = get_parser()
    show_root = attach_home_root(args.show_root)
    dataset = init_inpainting_eval_dataset(data_indexes_path=args.data_indexes_path, preprocess=False)
    data_iter = iter(dataset)
    length = len(dataset)
    progress_bar = ProgressBarTqdm(length)

    custom_power_paint = CustomPowerPaint(checkpoint_dir="model/PowerPaint-v1", version="ppt-v1")

    for i_data, data in enumerate(data_iter):
        image = data["image"].convert("RGB")
        mask = data["inpainting_mask"].convert("RGB")
        result = np.array(custom_power_paint(image=image, mask=mask)[0])
        image_np = np.array(image)
        mask_np = np.array(mask)[..., -1] / 255
        shows = [visual_mask(image_np, mask_np, stack_axis=1)[-1]]
        shows.append(result)
        shows = concat_differ_size(shows)
        write_im(os.path.join(show_root, 'visual', f"{i_data}.jpg"), shows[..., ::-1])
        write_im(os.path.join(show_root, 'result', f"{i_data}.jpg"), result[..., ::-1])
        pass

    # image_path = "/home/zhuyixing/datasets/inpainting_demo_v2/1.jpg"
    # mask_path = "/home/zhuyixing/datasets/inpainting_demo_v2/1.png"
    # image = obj_load(image_path)
    # mask = obj_load(mask_path)
    # mask = mask.convert("RGB")
    # custom_power_paint = CustomPowerPaint()
    # custom_power_paint(image=image, mask=mask)