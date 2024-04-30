# -*- coding: utf-8 -*-
import logging
import os
import os.path as osp
import pickle
import cv2
import mmcv
import numpy as np
import torch
import torch.utils.data as torchdata
from core.base_data_loader import Base_DatasetFromList
from core.utils.data_utils import (
    crop_resize_by_warp_affine,
    get_2d_coord_np,
    read_image_mmcv,
    xyz_to_region,
)
from core.utils.dataset_utils import (
    filter_empty_dets,
    filter_invalid_in_dataset_dicts,
    remove_anno_keys_dataset_dicts,
    flat_dataset_dicts,
    load_detections_into_dataset,
    my_build_batch_data_loader,
    trivial_batch_collator,
)
from core.utils.ssd_color_transform import ColorAugSSDTransform
from core.utils.depth_aug import add_noise_depth
from detectron2.data import MetadataCatalog, get_detection_dataset_dicts
from detectron2.utils.logger import log_first_n
from lib.pysixd import inout, misc
from lib.utils.mask_utils import cocosegm2mask, get_edge
from lib.vis_utils.image import grid_show
from dataset_factory import register_datasets
from data_loader_online import GDRN_Online_DatasetFromList
from data_loader import GDRN_DatasetFromList
from lib.utils.config_utils import try_get_key
import ref
import sys
from core.utils.my_distributed_sampler import TrainingSampler
from lib.vis_utils.image import grid_show, vis_bbox_opencv
from core.utils.data_utils import denormalize_image
from core.gdrn_modeling.engine.engine_utils import batch_data_train_online, get_renderer
from lib.egl_renderer.egl_renderer_v3 import EGLRenderer
import random


logger = logging.getLogger(__name__)

cfg = mmcv.Config.fromfile('configs/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo.py')


def vis_train_data(data, obj_names, cfg):
    for i, d in enumerate(data):
        #if i >= 1:
        #     continue
        full_img = mmcv.imread(d["file_name"], "color")
        #if "000009/rgb/000047.png" not in d["file_name"]:
        #     continue
        print(d["file_name"])
        im_H, im_W = full_img.shape[:2]
        roi_cls = d["roi_cls"]
        #if roi_cls not in [0]:
        #    continue
        bbox_center = d["bbox_center"]
        scale = d["scale"]
        x1 = max(min(bbox_center[0] - scale / 2, im_W), 0)
        x2 = max(min(bbox_center[0] + scale / 2, im_W), 0)
        y1 = max(min(bbox_center[1] - scale / 2, im_H), 0)
        y2 = max(min(bbox_center[1] + scale / 2, im_H), 0)
        full_img_vis = vis_bbox_opencv(full_img, np.array([x1, y1, x2, y2]), fmt="xyxy")

        bbox_ori = d["bbox"]
        full_img_bbox = vis_bbox_opencv(full_img, bbox_ori, fmt="xyxy")
        obj_name = obj_names[roi_cls]

        roi_img = d["roi_img"].numpy()
        roi_img = denormalize_image(roi_img, cfg).transpose(1, 2, 0).astype("uint8")

        roi_mask_trunc = d["roi_mask_trunc"].numpy().astype("bool")
        roi_mask_visib = d["roi_mask_visib"].numpy().astype("bool")
        roi_mask_obj = d["roi_mask_obj"].numpy().astype("bool")

        kernel = np.ones((3, 3), np.uint8)
        erode_mask_obj = cv2.erode(roi_mask_obj.astype("uint8"), kernel, iterations=1)

        roi_xyz = d["roi_xyz"].numpy().transpose(1, 2, 0)
        roi_xyz_show = get_emb_show(roi_xyz) * erode_mask_obj[:, :, None].astype("float32")

        coord2d = d["roi_coord_2d"].numpy().transpose(1, 2, 0)
        roi_h, roi_w = coord2d.shape[:2]
        zeros_1 = np.zeros((roi_h, roi_w, 1), dtype="float32")
        coord2d_3 = np.concatenate([zeros_1, get_emb_show(coord2d)], axis=2)

        # yapf: disable
        vis_imgs = [
            full_img_vis[:, :, [2, 1, 0]], full_img_bbox[:, :, [2, 1, 0]], roi_img[:, :, [2, 1, 0]],
            roi_mask_trunc * erode_mask_obj, roi_mask_visib*erode_mask_obj, roi_mask_obj*erode_mask_obj,
            roi_xyz_show,
            coord2d_3,
            coord2d[:, :, 0], coord2d[:, :, 1]
        ]
        titles = [
            "full_img", "ori_bbox", f"roi_img({obj_name})",
            "roi_mask_trunc",  "roi_mask_visib", "roi_mask_obj",
            "roi_xyz",
            "roi_coord2d",
            "roi_coord2d_x", "roi_coord2d_y"
        ]
        row = 3
        col = 4
        if "roi_region" in d:
            roi_region = d["roi_region"].numpy()  # (bh, bw)
            roi_region_3 = np.zeros((roi_h, roi_w, 3), dtype="float32")
            for region_id in range(256):
                # if region_id == 0:
                #     continue
                if region_id in roi_region:
                    for _c in range(3):
                        roi_region_3[:, :, _c][roi_region == region_id] = roi_xyz_show[:, :, _c][roi_region == region_id].mean()
            roi_region_3 = roi_region_3  * erode_mask_obj[:, :, None].astype("float32")
            vis_imgs.append(roi_region_3)
            titles.append("roi_region")
        if len(vis_imgs) > row * col:
            col += 1
        for _im, _name in zip(vis_imgs, titles):
            save_path = osp.join(cfg.OUTPUT_DIR, "vis", _name+'.png')
            print(f"Trying to save to {save_path} ...")
            mmcv.mkdir_or_exist(osp.dirname(save_path))
            if _im.shape[-1] == 3:
                _im = _im[:, :, [2,1,0]]
            if _im.max() < 1.1:
                _im = (_im * 255).astype("uint8")
            print(save_path)
            mmcv.imwrite(_im, save_path)
            print(f"Saved to {save_path}")

        grid_show(vis_imgs, titles, row=row, col=col)

def build_gdrn_train_loader(cfg, dataset_names):
    """A data loader is created by the following steps:

    1. Use the dataset names in config to query :class:`DatasetCatalog`, and obtain a list of dicts.
    2. Coordinate a random shuffle order shared among all processes (all GPUs)
    3. Each process spawn another few workers to process the dicts. Each worker will:
       * Map each metadata dict into another format to be consumed by the model.
       * Batch them by simply putting dicts into a list.

    The batched ``list[mapped_dict]`` is what this dataloader will yield.

    Args:
        cfg: the config

    Returns:
        an infinite iterator of training data
    """
    dataset_dicts = get_detection_dataset_dicts(
        dataset_names,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE if cfg.MODEL.KEYPOINT_ON else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
    )

    anno_keys_to_remove = try_get_key(cfg, "DATALOADER.REMOVE_ANNO_KEYS", default=[])
    if len(anno_keys_to_remove) > 0:
        dataset_dicts = remove_anno_keys_dataset_dicts(dataset_dicts, keys=anno_keys_to_remove)
        logger.warning(f"keys: {anno_keys_to_remove} removed from annotations")

    dataset_dicts = filter_invalid_in_dataset_dicts(dataset_dicts, visib_thr=cfg.DATALOADER.FILTER_VISIB_THR)

    if cfg.MODEL.POSE_NET.XYZ_ONLINE:
        dataset = GDRN_Online_DatasetFromList(cfg, split="train", lst=dataset_dicts, copy=False)
    else:
        dataset = GDRN_DatasetFromList(cfg, split="train", lst=dataset_dicts, copy=False)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN

    logger.info("Using training sampler {}".format(sampler_name))
    # TODO avoid if-else?
    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))
    return my_build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        persistent_workers=cfg.DATALOADER.PERSISTENT_WORKERS,
    )

from lm_pbr import register_with_name_cfg

register_with_name_cfg(cfg.DATASETS.TRAIN[0])
train_dset_meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
data_ref = ref.__dict__[train_dset_meta.ref_key]
train_obj_names = train_dset_meta.objs
renderer = get_renderer(cfg, data_ref, obj_names=train_obj_names, gpu_id=0)


train_dset_names = cfg.DATASETS.TRAIN
data_loader = build_gdrn_train_loader(cfg, train_dset_names)

data_loader_iter = iter(data_loader)
next_batch = next(iter(data_loader))


batch = batch_data_train_online(cfg, next_batch, renderer=renderer)
vis_train_data(batch, train_obj_names, cfg)