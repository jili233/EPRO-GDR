import copy
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.utils.solver_utils import build_optimizer_with_params
from detectron2.utils.events import get_event_storage
from mmcv.runner import load_checkpoint
import os
import time
#import pyprogressivex

from ..losses.coor_cross_entropy import CrossEntropyHeatmapLoss
from ..losses.l2_loss import L2Loss
from ..losses.mask_losses import weighted_ex_loss_probs, soft_dice_loss
from ..losses.pm_loss import PyPMLoss
from ..losses.rot_loss import angular_distance, rot_l2_loss
from .model_utils import (
    compute_mean_re_te,
    get_neck,
    get_geo_head,
    get_mask_prob,
    get_pnp_net,
    get_rot_mat,
    get_xyz_doublemask_region_out_dim,
)
from .pose_from_pred import pose_from_pred
from .pose_from_pred_centroid_z import pose_from_pred_centroid_z
from .pose_from_pred_centroid_z_abs import pose_from_pred_centroid_z_abs
from .net_factory import BACKBONES
#from core.utils.my_checkpoint import load_timm_pretrained
from core.gdrn_modeling.epropnp.lib.ops.pnp.epropnp import EProPnP6DoF
from core.gdrn_modeling.epropnp.lib.ops.pnp.camera import PerspectiveCamera
from core.gdrn_modeling.epropnp.lib.ops.pnp.levenberg_marquardt import LMSolver, RSLMSolver
from core.gdrn_modeling.epropnp.lib.ops.pnp.cost_fun import AdaptiveHuberPnPCost
from core.gdrn_modeling.epropnp.lib.models.monte_carlo_pose_loss import MonteCarloPoseLoss
from core.gdrn_modeling.epropnp.lib.ops.rotation_conversions import matrix_to_quaternion
import cv2
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

class GDRN_DoubleMask(nn.Module):
    def __init__(self, cfg, backbone, geo_head_net, epropnp_head=None, neck=None, pnp_net=None):
        super().__init__()
        assert cfg.MODEL.POSE_NET.NAME == "GDRN_double_mask", cfg.MODEL.POSE_NET.NAME
        self.backbone = backbone
        self.neck = neck

        self.geo_head_net = geo_head_net
        self.epropnp_head = epropnp_head

        self.cfg = cfg
        self.xyz_out_dim, self.mask_out_dim, self.region_out_dim = get_xyz_doublemask_region_out_dim(cfg)

        # uncertainty multi-task loss weighting
        # https://github.com/Hui-Li/multi-task-learning-example-PyTorch/blob/master/multi-task-learning-example-PyTorch.ipynb
        # a = log(sigma^2)
        # L*exp(-a) + a  or  L*exp(-a) + log(1+exp(a))
        # self.log_vars = nn.Parameter(torch.tensor([0, 0], requires_grad=True, dtype=torch.float32).cuda())
        # yapf: disable
        if cfg.MODEL.POSE_NET.USE_MTL:
            self.loss_names = [
                "mask", "coor_x", "coor_y", "coor_z", "coor_x_bin", "coor_y_bin", "coor_z_bin", "region",
                "PM_R", "PM_xy", "PM_z", "PM_xy_noP", "PM_z_noP", "PM_T", "PM_T_noP",
                "centroid", "z", "trans_xy", "trans_z", "trans_LPnP", "rot", "bind", "monte_carlo",
            ]
            for loss_name in self.loss_names:
                self.register_parameter(
                    f"log_var_{loss_name}", nn.Parameter(torch.tensor([0.0], requires_grad=True, dtype=torch.float32))
                )
                
        self.mc_loss_fn = MonteCarloPoseLoss(init_norm_factor=1.0, momentum=0.01)
    
        # yapf: enable
        self.pnp_net_train = EProPnP6DoF(
            mc_samples=cfg.MODEL.POSE_NET.PNP_NET.NUM_POINTS,
            num_iter=4,
            solver=LMSolver(
                dof=6,
                num_iter=5,
                init_solver=RSLMSolver(
                    dof=6,
                    num_points=16,
                    num_proposals=4,
                    num_iter=3))).cuda()
        
        self.pnp_net_test = EProPnP6DoF(
            mc_samples=cfg.MODEL.POSE_NET.PNP_NET.NUM_POINTS,
            num_iter=4,
            solver=LMSolver(
                dof=6,
                num_iter=10)).cuda()

    def forward(
        self,
        x,
        resize_ratios,
        roi_extents,
        gt_ego_rot=None,
        gt_trans=None,
        gt_xyz=None,
        gt_xyz_bin=None,
        gt_mask_trunc=None,
        gt_mask_visib=None,
        gt_mask_obj=None,
        gt_mask_full=None,
        gt_region=None,
        gt_points=None,
        sym_infos=None,
        gt_trans_ratio=None,
        roi_classes=None,
        roi_coord_2d=None,
        roi_coord_2d_rel=None,
        roi_cams=None,
        roi_centers=None,
        roi_whs=None,
        do_loss=False,
    ):
        cfg = self.cfg
        net_cfg = cfg.MODEL.POSE_NET
        g_head_cfg = net_cfg.GEO_HEAD
        pnp_net_cfg = net_cfg.PNP_NET
        loss_cfg = net_cfg.LOSS_CFG

        device = x.device
        bs = x.shape[0]
        num_classes = net_cfg.NUM_CLASSES
        out_res = net_cfg.OUTPUT_RES

        # x.shape [bs, 3, 256, 256]
        conv_feat = self.backbone(x)  # [bs, c, 8, 8]
        if self.neck is not None:
            conv_feat = self.neck(conv_feat)
        vis_mask, full_mask, coor_x, coor_y, coor_z, region, w2d, scale = self.geo_head_net(conv_feat)

        if g_head_cfg.XYZ_CLASS_AWARE:
            assert roi_classes is not None
            coor_x = coor_x.view(bs, num_classes, self.xyz_out_dim // 3, out_res, out_res)
            coor_x = coor_x[torch.arange(bs).to(device), roi_classes]
            coor_y = coor_y.view(bs, num_classes, self.xyz_out_dim // 3, out_res, out_res)
            coor_y = coor_y[torch.arange(bs).to(device), roi_classes]
            coor_z = coor_z.view(bs, num_classes, self.xyz_out_dim // 3, out_res, out_res)
            coor_z = coor_z[torch.arange(bs).to(device), roi_classes]

        if g_head_cfg.MASK_CLASS_AWARE:
            assert roi_classes is not None
            vis_mask = vis_mask.view(bs, num_classes, self.mask_out_dim // 2, out_res, out_res)
            vis_mask = vis_mask[torch.arange(bs).to(device), roi_classes]
            full_mask = full_mask.view(bs, num_classes, self.mask_out_dim // 2, out_res, out_res)
            full_mask = full_mask[torch.arange(bs).to(device), roi_classes]

        if g_head_cfg.REGION_CLASS_AWARE:
            assert roi_classes is not None
            region = region.view(bs, num_classes, self.region_out_dim, out_res, out_res)
            region = region[torch.arange(bs).to(device), roi_classes]

        # -----------------------------------------------
        # get rot and trans from pnp_net
        # NOTE: use softmax for bins (the last dim is bg)
        if coor_x.shape[1] > 1 and coor_y.shape[1] > 1 and coor_z.shape[1] > 1:
            coor_x_softmax = F.softmax(coor_x[:, :-1, :, :], dim=1)
            coor_y_softmax = F.softmax(coor_y[:, :-1, :, :], dim=1)
            coor_z_softmax = F.softmax(coor_z[:, :-1, :, :], dim=1)
            coor_feat = torch.cat([coor_x_softmax, coor_y_softmax, coor_z_softmax], dim=1)
        else:
            coor_feat = torch.cat([coor_x, coor_y, coor_z], dim=1)  # BCHW
        

        if pnp_net_cfg.WITH_2D_COORD:
            if pnp_net_cfg.COORD_2D_TYPE == "rel":
                assert roi_coord_2d_rel is not None
                coor_feat = torch.cat([coor_feat, roi_coord_2d_rel], dim=1)
            else:  # default is abs not rel
                assert roi_coord_2d is not None
                x3d = coor_feat
                # x2d will be calculated the same way as EPro-PnP
                # x2d = roi_coord_2d

        # NOTE: for region, the 1st dim is bg
        region_softmax = F.softmax(region[:, 1:, :, :], dim=1)

        # -----------------------------------------------
        # https://github.com/tjiiv-cprg/EPro-PnP-v2/blob/main/EPro-PnP-6DoF_v2/lib/train.py
        # preparing data for epropnp, x3d 
        # scale x3d to the original size, unit is meter.
        x3d = (x3d-0.5) * roi_extents[..., None, None]  # (bs, 3, h, w)
        scales = net_cfg.OUTPUT_RES / resize_ratios
        wh_begin = roi_centers.to(torch.int64) - torch.stack((scales,scales), dim=1) / 2  # (n, 2)

        # preparing x2d
        x2d_list = []
        wh_unit_list = []
        for i in range(bs):
            wh_unit = 1 / resize_ratios[i]
            wh_arange_current = torch.arange(net_cfg.OUTPUT_RES, dtype=torch.float).to(device)
            yy, xx = torch.meshgrid(wh_arange_current, wh_arange_current)
            x2d = torch.stack((wh_begin[i, 0].to(device) + xx.to(device) * wh_unit, wh_begin[i, 1].to(device) + yy.to(device) * wh_unit), dim=-1)
            x2d_list.append(x2d)
            wh_unit_list.append(wh_unit)
        x2d = torch.stack(x2d_list, dim=0)
        wh_unit = torch.stack(wh_unit_list, dim=0)

        # shape from (bs, dim, 64, 64) to (bs, pts, dim)
        w2d = w2d.permute(0, 2, 3, 1)
        w2d = w2d.reshape(bs, -1, 2)
        x3d = x3d.flatten(2).transpose(-1, -2)
        x2d = x2d.reshape(bs, net_cfg.OUTPUT_RES**2, 2)

        # diff. between EPro-PnP-v1 and v2
        resize_ratios_reshape = resize_ratios.clone()
        resize_ratios_reshape = resize_ratios_reshape.unsqueeze(1).unsqueeze(2)
        resize_ratios_reshape = resize_ratios_reshape.expand_as(w2d)
        scale_reshape = scale.clone()
        scale_reshape = scale_reshape.unsqueeze(1)
        scale_reshape = scale_reshape.expand_as(w2d)
        w2d = w2d.softmax(dim=1) * resize_ratios_reshape * scale_reshape

        # sampling points
        sample_pts = [np.random.choice(net_cfg.OUTPUT_RES**2, size=pnp_net_cfg.NUM_POINTS, replace=False) for _ in range(bs)]
        sample_inds = x2d.new_tensor(sample_pts, dtype=torch.int64)
        
        if do_loss: # during the training we use the gt pose to initialize.
            R_quats_gt = matrix_to_quaternion(gt_ego_rot)
            T_vectors_gt = gt_trans
            pose_init = torch.cat((T_vectors_gt, R_quats_gt), dim=-1)  # (n, 7)

            x3d_sampled = x3d[torch.arange(x3d.size(0))[:, None], sample_inds]
            x2d_sampled = x2d[torch.arange(x2d.size(0))[:, None], sample_inds]
            w2d_sampled = w2d[torch.arange(w2d.size(0))[:, None], sample_inds]

            allowed_border = 30 * wh_unit
            camera = PerspectiveCamera(
            cam_mats=roi_cams[0].expand(bs, -1, -1),
            z_min=0.01,
            lb=wh_begin - allowed_border[:, None],
            ub=wh_begin + (net_cfg.OUTPUT_RES - 1) * wh_unit[:, None] + allowed_border[:, None]
            )
            cost_fun = AdaptiveHuberPnPCost(
                relative_delta=0.1)
            cost_fun.set_param(x2d_sampled, w2d_sampled)

            _, _, pose_opt_plus, _, pose_sample_logweights, cost_tgt = self.pnp_net_train.monte_carlo_forward(
                x3d_sampled,
                x2d_sampled,
                w2d_sampled,
                camera,
                cost_fun,
                pose_init=pose_init.to(device),
                force_init_solve=True,
                with_pose_opt_plus=True)
            
            # monte carlo loss
            loss_mc = self.mc_loss_fn(
                pose_sample_logweights.to(device),
                cost_tgt.to(device),
                scale.detach().mean().to(device)
            )
            
            norm_factor = self.mc_loss_fn.norm_factor.item()
            # print("norm_factor: ", norm_factor)

            pred_t_, R_quats = pose_opt_plus.split([3, 4], dim=-1)  # (n, [3, 4])
            pred_rot_ = R.from_quat(R_quats[:, [1, 2, 3, 0]].detach().cpu().numpy()).as_matrix()  # (n, 3, 3)
            pred_rot_m = torch.tensor(pred_rot_)
            pred_t_ = torch.tensor(pred_t_)

            pred_ego_rot = pred_rot_m.float().to(device)
            pred_trans = pred_t_.float().to(device)
            
        else:
            x2d_np = x2d.clone()
            x2d_np = x2d_np.cpu().numpy()
            x3d_np = x3d.clone()
            x3d_np = x3d_np.cpu().numpy()
            w2d = w2d * wh_unit[:, None, None]

            # calculate the mask based on the w2d
            if cfg.TEST.USE_W2D_MASK:
                assert pnp_net_cfg.INIT_THRESHOLD is not None
                pred_conf = w2d.mean(axis=2).squeeze()
                pred_conf = pred_conf.cpu().detach().numpy()
                valid_mask_list = []
                for pred_conf_ in pred_conf:
                    valid_mask = pred_conf_ >= np.quantile(pred_conf_, pnp_net_cfg.INIT_THRESHOLD, keepdims=True)[..., None]
                    valid_mask_squeezed = valid_mask.squeeze()
                    valid_mask_list.append(torch.tensor(valid_mask_squeezed))
                valid_mask = torch.stack(valid_mask_list, dim=0)
                roi_cams = roi_cams.cpu().numpy()
                valid_mask = valid_mask.cpu().numpy()
            
            # calculate the mask based on the visible mask predicted
            if cfg.TEST.USE_PRED_VIS_MASK:
                valid_mask_list = []
                for i in range(vis_mask.shape[0]):
                    vis_mask_1d = vis_mask[i].reshape(-1)
                    vis_mask_bool = vis_mask_1d > 0.5
                    valid_mask_list.append(torch.tensor(vis_mask_bool))
                valid_mask = torch.stack(valid_mask_list, dim=0)
                roi_cams = roi_cams.cpu().numpy()
                valid_mask = valid_mask.cpu().numpy()

            dist_coeffs = np.zeros((4, 1), dtype=np.float64)
            x2d_filtered = []
            x3d_filtered = []
            w2d_filtered = []
            R_quats = []
            T_vectors = []
            w2d = w2d / wh_unit[:, None, None]
            for x2d_np_, x3d_np_, w2d_, mask_np_, roi_cams_ in zip(x2d_np, x3d_np, w2d, valid_mask, roi_cams):
                _, R_vector, T_vector, _ = cv2.solvePnPRansac(
                    x3d_np_[mask_np_], x2d_np_[mask_np_], roi_cams_, dist_coeffs)
                q = R.from_rotvec(R_vector.reshape(-1)).as_quat()[[3, 0, 1, 2]]
                R_quats.append(q)
                T_vectors.append(T_vector.reshape(-1))
            
            # Progressive-X
            # https://github.com/suyz526/ZebraPose/blob/06b53d2aab1f5e08ffc927b6c5a02e9913cb13f8/zebrapose/binary_code_helper/CNN_output_to_pose.py
            # for x2d_np_, x3d_np_, w2d_, mask_np_, roi_cams_ in zip(x2d_np, x3d_np, w2d, valid_mask, roi_cams):
            #     coord_2d = np.ascontiguousarray(x2d_np_[mask_np_].astype(np.float64))
            #     coord_3d = np.ascontiguousarray(x3d_np_[mask_np_].astype(np.float64))
            #     intrinsic_matrix = np.ascontiguousarray(roi_cams_.astype(np.float64))

            #     pose_ests, label = pyprogressivex.find6DPoses(
            #         x1y1=coord_2d,
            #         x2y2z2=coord_3d,
            #         K=intrinsic_matrix,
            #         threshold=2,
            #         neighborhood_ball_radius=20,
            #         spatial_coherence_weight=0.1,
            #         maximum_tanimoto_similarity=0.9,
            #         max_iters=400,
            #         minimum_point_number=6,
            #         maximum_model_number=1
            #     )

            #     if pose_ests.shape[0] != 0:
            #         rot = pose_ests[0:3, :3]
            #         tvecs = pose_ests[0:3, 3].reshape(-1)
            #         q = R.from_matrix(rot).as_quat()[[3, 0, 1, 2]]  # 转换为四元数
            #         R_quats.append(q)
            #         T_vectors.append(tvecs)
            #     else:
            #         rot = np.zeros((3,3))
            #         tvecs = np.zeros((3,1))
            #         success = False
            #         continue

            pose_init_test = torch.cat((torch.tensor(T_vectors), torch.tensor(R_quats)), dim=-1).float().to(device)  # (n, 7)

            camera = PerspectiveCamera(
            cam_mats=torch.tensor(roi_cams[0]).to(device),
            z_min=0.01,
            )

            cost_fun = AdaptiveHuberPnPCost(
                relative_delta=0.1)
        
            pose_opt_list = []

            cost_fun.set_param(x2d, w2d)
            single_pose_opt, _, _, _, _, _ = self.pnp_net_test.monte_carlo_forward(
                x3d,
                x2d,
                w2d,
                camera,
                cost_fun,
                pose_init=pose_init_test,
                force_init_solve=False,
                with_pose_opt_plus=False
            )
            pose_opt_list.append(single_pose_opt)
            pose_opt = torch.cat(pose_opt_list, dim=0)
            pred_t_, R_quats = pose_opt.split([3, 4], dim=-1)  # (n, [3, 4])
            pred_rot_ = R.from_quat(R_quats[:, [1, 2, 3, 0]].detach().cpu().numpy()).as_matrix()  # (n, 3, 3)
            pred_rot_m = torch.tensor(pred_rot_)
            pred_t_ = torch.tensor(pred_t_)

            pred_ego_rot = pred_rot_m.float().to(device)
            pred_trans = pred_t_.float().to(device)

        if not do_loss:  # test
            out_dict = {"rot": pred_ego_rot, "trans": pred_trans}
            if cfg.TEST.USE_PNP or cfg.TEST.SAVE_RESULTS_ONLY or cfg.TEST.USE_DEPTH_REFINE:
                # TODO: move the pnp/ransac inside forward
                out_dict.update(
                    {
                        "x": x,
                        "mask": vis_mask,
                        "full_mask": full_mask,
                        "coor_x": coor_x,
                        "coor_y": coor_y,
                        "coor_z": coor_z,
                        "region": region_softmax,
                        "w2d": w2d,
                        "x3d": x3d,
                        "x2d": x2d,
                        "conf_mask": valid_mask,
                        "x2d_filtered": x2d_filtered,
                        "x3d_filtered": x3d_filtered,
                        "w2d_filtered": w2d_filtered,
                    }
                )
        else:
            out_dict = {}
            assert (
                (gt_xyz is not None)
                and (gt_trans is not None)
                and (gt_trans_ratio is not None)
                and (gt_region is not None)
            )
            mean_re, mean_te = compute_mean_re_te(pred_trans, pred_rot_m, gt_trans, gt_ego_rot)
            vis_dict = {
                "vis/error_R": mean_re,
                "vis/error_t": mean_te * 100,  # cm
                "vis/error_tx": np.abs(pred_trans[0, 0].detach().item() - gt_trans[0, 0].detach().item()) * 100,  # cm
                "vis/error_ty": np.abs(pred_trans[0, 1].detach().item() - gt_trans[0, 1].detach().item()) * 100,  # cm
                "vis/error_tz": np.abs(pred_trans[0, 2].detach().item() - gt_trans[0, 2].detach().item()) * 100,  # cm
                "vis/tx_pred": pred_trans[0, 0].detach().item(),
                "vis/ty_pred": pred_trans[0, 1].detach().item(),
                "vis/tz_pred": pred_trans[0, 2].detach().item(),
                "vis/tx_net": pred_t_[0, 0].detach().item(),
                "vis/ty_net": pred_t_[0, 1].detach().item(),
                "vis/tz_net": pred_t_[0, 2].detach().item(),
                "vis/tx_gt": gt_trans[0, 0].detach().item(),
                "vis/ty_gt": gt_trans[0, 1].detach().item(),
                "vis/tz_gt": gt_trans[0, 2].detach().item(),
                "vis/tx_rel_gt": gt_trans_ratio[0, 0].detach().item(),
                "vis/ty_rel_gt": gt_trans_ratio[0, 1].detach().item(),
                "vis/tz_rel_gt": gt_trans_ratio[0, 2].detach().item(),
            }

            loss_dict = self.gdrn_loss(
                cfg=self.cfg,
                out_mask_vis=vis_mask,
                out_mask_full=full_mask,
                gt_mask_trunc=gt_mask_trunc,
                gt_mask_visib=gt_mask_visib,
                gt_mask_obj=gt_mask_obj,
                gt_mask_full=gt_mask_full,
                out_x=coor_x,
                out_y=coor_y,
                out_z=coor_z,
                gt_xyz=gt_xyz,
                gt_xyz_bin=gt_xyz_bin,
                out_region=region,
                gt_region=gt_region,
                out_trans=pred_trans,
                gt_trans=gt_trans,
                out_rot=pred_ego_rot,
                gt_rot=gt_ego_rot,
                out_centroid=pred_t_[:, :2],  # TODO: get these from trans head
                out_trans_z=pred_t_[:, 2],
                gt_trans_ratio=gt_trans_ratio,
                gt_points=gt_points,
                sym_infos=sym_infos,
                extents=roi_extents,
                loss_mc = loss_mc,
            )
            if net_cfg.USE_MTL:
                for _name in self.loss_names:
                    if f"loss_{_name}" in loss_dict:
                        vis_dict[f"vis_lw/{_name}"] = torch.exp(-getattr(self, f"log_var_{_name}")).detach().item()
            for _k, _v in vis_dict.items():
                if "vis/" in _k or "vis_lw/" in _k:
                    if isinstance(_v, torch.Tensor):
                        _v = _v.item()
                    vis_dict[_k] = _v
            storage = get_event_storage()
            storage.put_scalars(**vis_dict)

            return out_dict, loss_dict
        return out_dict

    def gdrn_loss(
        self,
        cfg,
        out_mask_vis,
        out_mask_full,
        gt_mask_trunc,
        gt_mask_visib,
        gt_mask_obj,
        gt_mask_full,
        out_x,
        out_y,
        out_z,
        gt_xyz,
        gt_xyz_bin,
        out_region,
        gt_region,
        loss_mc,
        out_rot=None,
        gt_rot=None,
        out_trans=None,
        gt_trans=None,
        out_centroid=None,
        out_trans_z=None,
        gt_trans_ratio=None,
        gt_points=None,
        sym_infos=None,
        extents=None,
    ):
        net_cfg = cfg.MODEL.POSE_NET
        g_head_cfg = net_cfg.GEO_HEAD
        pnp_net_cfg = net_cfg.PNP_NET
        loss_cfg = net_cfg.LOSS_CFG
        bs = out_mask_vis.shape[0]
        device = out_mask_vis.device

        loss_dict = {}

        gt_masks = {"trunc": gt_mask_trunc, "visib": gt_mask_visib, "obj": gt_mask_obj, "full": gt_mask_full}

        # xyz loss ----------------------------------
        if not g_head_cfg.FREEZE:
            xyz_loss_type = loss_cfg.XYZ_LOSS_TYPE
            gt_mask_xyz = gt_masks[loss_cfg.XYZ_LOSS_MASK_GT]
            if xyz_loss_type == "L1":
                loss_func = nn.L1Loss(reduction="sum")
                loss_dict["loss_coor_x"] = loss_func(
                    out_x * gt_mask_xyz[:, None], gt_xyz[:, 0:1] * gt_mask_xyz[:, None]
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
                loss_dict["loss_coor_y"] = loss_func(
                    out_y * gt_mask_xyz[:, None], gt_xyz[:, 1:2] * gt_mask_xyz[:, None]
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
                loss_dict["loss_coor_z"] = loss_func(
                    out_z * gt_mask_xyz[:, None], gt_xyz[:, 2:3] * gt_mask_xyz[:, None]
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
            elif xyz_loss_type == "CE_coor":
                gt_xyz_bin = gt_xyz_bin.long()
                loss_func = CrossEntropyHeatmapLoss(reduction="sum", weight=None)  # g_head_cfg.XYZ_BIN+1
                loss_dict["loss_coor_x"] = loss_func(
                    out_x * gt_mask_xyz[:, None], gt_xyz_bin[:, 0] * gt_mask_xyz.long()
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
                loss_dict["loss_coor_y"] = loss_func(
                    out_y * gt_mask_xyz[:, None], gt_xyz_bin[:, 1] * gt_mask_xyz.long()
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
                loss_dict["loss_coor_z"] = loss_func(
                    out_z * gt_mask_xyz[:, None], gt_xyz_bin[:, 2] * gt_mask_xyz.long()
                ) / gt_mask_xyz.sum().float().clamp(min=1.0)
            else:
                raise NotImplementedError(f"unknown xyz loss type: {xyz_loss_type}")
            loss_dict["loss_coor_x"] *= loss_cfg.XYZ_LW
            loss_dict["loss_coor_y"] *= loss_cfg.XYZ_LW
            loss_dict["loss_coor_z"] *= loss_cfg.XYZ_LW

        # mask loss ----------------------------------
        if not g_head_cfg.FREEZE:
            mask_loss_type = loss_cfg.MASK_LOSS_TYPE
            gt_mask = gt_masks[loss_cfg.MASK_LOSS_GT]
            if mask_loss_type == "L1":
                loss_dict["loss_mask"] = nn.L1Loss(reduction="mean")(out_mask_vis[:, 0, :, :], gt_mask)
            elif mask_loss_type == "BCE":
                loss_dict["loss_mask"] = nn.BCEWithLogitsLoss(reduction="mean")(out_mask_vis[:, 0, :, :], gt_mask)
            elif mask_loss_type == "RW_BCE":
                loss_dict["loss_mask"] = weighted_ex_loss_probs(
                    torch.sigmoid(out_mask_vis[:, 0, :, :]), gt_mask, weight=None
                )
            elif mask_loss_type == "dice":
                loss_dict["loss_mask"] = soft_dice_loss(
                    torch.sigmoid(out_mask_vis[:, 0, :, :]), gt_mask, eps=0.002, reduction="mean"
                )
            elif mask_loss_type == "CE":
                loss_dict["loss_mask"] = nn.CrossEntropyLoss(reduction="mean")(out_mask_vis, gt_mask.long())
            else:
                raise NotImplementedError(f"unknown mask loss type: {mask_loss_type}")
            loss_dict["loss_mask"] *= loss_cfg.MASK_LW

        if not g_head_cfg.FREEZE and loss_cfg.FULL_MASK_LW > 0:
            assert gt_mask_full is not None
            full_mask_loss_type = loss_cfg.FULL_MASK_LOSS_TYPE
            if full_mask_loss_type == "L1":
                loss_dict["loss_mask_full"] = nn.L1Loss(reduction="mean")(out_mask_full[:, 0, :, :], gt_mask_full)
            elif full_mask_loss_type == "BCE":
                loss_dict["loss_mask_full"] = nn.BCEWithLogitsLoss(reduction="mean")(
                    out_mask_full[:, 0, :, :], gt_mask_full
                )
            elif full_mask_loss_type == "RW_BCE":
                loss_dict["loss_mask_full"] = weighted_ex_loss_probs(
                    torch.sigmoid(out_mask_full[:, 0, :, :]), gt_mask_full, weight=None
                )
            elif full_mask_loss_type == "dice":
                loss_dict["loss_mask_full"] = soft_dice_loss(
                    torch.sigmoid(out_mask_full[:, 0, :, :]), gt_mask_full, eps=0.002, reduction="mean"
                )
            elif full_mask_loss_type == "CE":
                loss_dict["loss_mask_full"] = nn.CrossEntropyLoss(reduction="mean")(out_mask_full, gt_mask_full.long())
            else:
                raise NotImplementedError(f"unknown mask loss type: {mask_loss_type}")
            loss_dict["loss_mask_full"] *= loss_cfg.FULL_MASK_LW

        # roi region loss --------------------
        if not g_head_cfg.FREEZE:
            region_loss_type = loss_cfg.REGION_LOSS_TYPE
            gt_mask_region = gt_masks[loss_cfg.REGION_LOSS_MASK_GT]
            if region_loss_type == "CE":
                gt_region = gt_region.long()
                loss_func = nn.CrossEntropyLoss(reduction="sum", weight=None)  # g_head_cfg.XYZ_BIN+1
                loss_dict["loss_region"] = loss_func(
                    out_region * gt_mask_region[:, None], gt_region * gt_mask_region.long()
                ) / gt_mask_region.sum().float().clamp(min=1.0)
            else:
                raise NotImplementedError(f"unknown region loss type: {region_loss_type}")
            loss_dict["loss_region"] *= loss_cfg.REGION_LW

        # point matching loss ---------------
        if loss_cfg.PM_LW > 0:
            assert (gt_points is not None) and (gt_trans is not None) and (gt_rot is not None)
            loss_func = PyPMLoss(
                loss_type=loss_cfg.PM_LOSS_TYPE,
                beta=loss_cfg.PM_SMOOTH_L1_BETA,
                reduction="mean",
                loss_weight=loss_cfg.PM_LW,
                norm_by_extent=loss_cfg.PM_NORM_BY_EXTENT,
                symmetric=loss_cfg.PM_LOSS_SYM,
                disentangle_t=loss_cfg.PM_DISENTANGLE_T,
                disentangle_z=loss_cfg.PM_DISENTANGLE_Z,
                t_loss_use_points=loss_cfg.PM_T_USE_POINTS,
                r_only=loss_cfg.PM_R_ONLY,
            )
            loss_pm_dict = loss_func(
                pred_rots=out_rot,
                gt_rots=gt_rot,
                points=gt_points,
                pred_transes=out_trans,
                gt_transes=gt_trans,
                extents=extents,
                sym_infos=sym_infos,
            )
            loss_dict.update(loss_pm_dict)

        # rot_loss ----------
        if loss_cfg.ROT_LW > 0:
            if loss_cfg.ROT_LOSS_TYPE == "angular":
                loss_dict["loss_rot"] = angular_distance(out_rot, gt_rot)
            elif loss_cfg.ROT_LOSS_TYPE == "L2":
                loss_dict["loss_rot"] = rot_l2_loss(out_rot, gt_rot)
            else:
                raise ValueError(f"Unknown rot loss type: {loss_cfg.ROT_LOSS_TYPE}")
            loss_dict["loss_rot"] *= loss_cfg.ROT_LW

        # centroid loss -------------
        if loss_cfg.CENTROID_LW > 0:
            assert (
                pnp_net_cfg.TRANS_TYPE == "centroid_z"
            ), "centroid loss is only valid for predicting centroid2d_rel_delta"

            if loss_cfg.CENTROID_LOSS_TYPE == "L1":
                loss_dict["loss_centroid"] = nn.L1Loss(reduction="mean")(out_centroid, gt_trans[:, :2])
            elif loss_cfg.CENTROID_LOSS_TYPE == "L2":
                loss_dict["loss_centroid"] = L2Loss(reduction="mean")(out_centroid, gt_trans[:, :2])
            elif loss_cfg.CENTROID_LOSS_TYPE == "MSE":
                loss_dict["loss_centroid"] = nn.MSELoss(reduction="mean")(out_centroid, gt_trans[:, :2])
            else:
                raise ValueError(f"Unknown centroid loss type: {loss_cfg.CENTROID_LOSS_TYPE}")
            loss_dict["loss_centroid"] *= loss_cfg.CENTROID_LW

        # z loss ------------------
        if loss_cfg.Z_LW > 0:
            z_type = pnp_net_cfg.Z_TYPE
            if z_type == "REL":
                gt_z = gt_trans_ratio[:, 2]
            elif z_type == "ABS":
                gt_z = gt_trans[:, 2]
            else:
                raise NotImplementedError

            z_loss_type = loss_cfg.Z_LOSS_TYPE
            if z_loss_type == "L1":
                loss_dict["loss_z"] = nn.L1Loss(reduction="mean")(out_trans_z, gt_z)
            elif z_loss_type == "L2":
                loss_dict["loss_z"] = L2Loss(reduction="mean")(out_trans_z, gt_z)
            elif z_loss_type == "MSE":
                loss_dict["loss_z"] = nn.MSELoss(reduction="mean")(out_trans_z, gt_z)
            else:
                raise ValueError(f"Unknown z loss type: {z_loss_type}")
            loss_dict["loss_z"] *= loss_cfg.Z_LW

        # trans loss ------------------
        if loss_cfg.TRANS_LW > 0:
            if loss_cfg.TRANS_LOSS_DISENTANGLE:
                # NOTE: disentangle xy/z
                if loss_cfg.TRANS_LOSS_TYPE == "L1":
                    loss_dict["loss_trans_xy"] = nn.L1Loss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = nn.L1Loss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                elif loss_cfg.TRANS_LOSS_TYPE == "L2":
                    loss_dict["loss_trans_xy"] = L2Loss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = L2Loss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                elif loss_cfg.TRANS_LOSS_TYPE == "MSE":
                    loss_dict["loss_trans_xy"] = nn.MSELoss(reduction="mean")(out_trans[:, :2], gt_trans[:, :2])
                    loss_dict["loss_trans_z"] = nn.MSELoss(reduction="mean")(out_trans[:, 2], gt_trans[:, 2])
                else:
                    raise ValueError(f"Unknown trans loss type: {loss_cfg.TRANS_LOSS_TYPE}")
                loss_dict["loss_trans_xy"] *= loss_cfg.TRANS_LW
                loss_dict["loss_trans_z"] *= loss_cfg.TRANS_LW
            else:
                if loss_cfg.TRANS_LOSS_TYPE == "L1":
                    loss_dict["loss_trans_LPnP"] = nn.L1Loss(reduction="mean")(out_trans, gt_trans)
                elif loss_cfg.TRANS_LOSS_TYPE == "L2":
                    loss_dict["loss_trans_LPnP"] = L2Loss(reduction="mean")(out_trans, gt_trans)

                elif loss_cfg.TRANS_LOSS_TYPE == "MSE":
                    loss_dict["loss_trans_LPnP"] = nn.MSELoss(reduction="mean")(out_trans, gt_trans)
                else:
                    raise ValueError(f"Unknown trans loss type: {loss_cfg.TRANS_LOSS_TYPE}")
                loss_dict["loss_trans_LPnP"] *= loss_cfg.TRANS_LW

        # bind loss (R^T@t)
        if loss_cfg.get("BIND_LW", 0.0) > 0.0:
            pred_bind = torch.bmm(out_rot.permute(0, 2, 1), out_trans.view(-1, 3, 1)).view(-1, 3)
            gt_bind = torch.bmm(gt_rot.permute(0, 2, 1), gt_trans.view(-1, 3, 1)).view(-1, 3)
            if loss_cfg.BIND_LOSS_TYPE == "L1":
                loss_dict["loss_bind"] = nn.L1Loss(reduction="mean")(pred_bind, gt_bind)
            elif loss_cfg.BIND_LOSS_TYPE == "L2":
                loss_dict["loss_bind"] = L2Loss(reduction="mean")(pred_bind, gt_bind)
            elif loss_cfg.CENTROID_LOSS_TYPE == "MSE":
                loss_dict["loss_bind"] = nn.MSELoss(reduction="mean")(pred_bind, gt_bind)
            else:
                raise ValueError(f"Unknown bind loss (R^T@t) type: {loss_cfg.BIND_LOSS_TYPE}")
            loss_dict["loss_bind"] *= loss_cfg.BIND_LW
        
        # monte carlo loss
        if loss_cfg.get("MC_LW", 0.0) > 0.0:
            loss_dict["loss_monte_carlo"] = loss_mc
            loss_dict["loss_monte_carlo"] *= loss_cfg.MC_LW

        if net_cfg.USE_MTL:
            for _k in loss_dict:
                _name = _k.replace("loss_", "log_var_")
                cur_log_var = getattr(self, _name)
                loss_dict[_k] = loss_dict[_k] * torch.exp(-cur_log_var) + torch.log(1 + torch.exp(cur_log_var))
        return loss_dict

def build_model_optimizer(cfg, is_test=False):
    net_cfg = cfg.MODEL.POSE_NET
    backbone_cfg = net_cfg.BACKBONE

    params_lr_list = []
    # backbone
    init_backbone_args = copy.deepcopy(backbone_cfg.INIT_CFG)
    backbone_type = init_backbone_args.pop("type")
    if "timm/" in backbone_type or "tv/" in backbone_type:
        init_backbone_args["model_name"] = backbone_type.split("/")[-1]

    backbone = BACKBONES[backbone_type](**init_backbone_args)

    if backbone_cfg.FREEZE:
        for param in backbone.parameters():
            with torch.no_grad():
                param.requires_grad = False
    else:
        params_lr_list.append(
            {"params": filter(lambda p: p.requires_grad, backbone.parameters()), "lr": float(cfg.SOLVER.BASE_LR)}
        )

    # neck --------------------------------
    neck, neck_params = get_neck(cfg)
    params_lr_list.extend(neck_params)

    # geo head -----------------------------------------------------
    geo_head, geo_head_params = get_geo_head(cfg)
    params_lr_list.extend(geo_head_params)

    # build model
    model = GDRN_DoubleMask(cfg, backbone, neck=neck, geo_head_net=geo_head)
    if net_cfg.USE_MTL:
        params_lr_list.append(
            {
                "params": filter(
                    lambda p: p.requires_grad,
                    [_param for _name, _param in model.named_parameters() if "log_var" in _name],
                ),
                "lr": float(cfg.SOLVER.BASE_LR),
            }
        )

    # get optimizer
    if is_test:
        optimizer = None
    else:
        optimizer = build_optimizer_with_params(cfg, params_lr_list)

    if cfg.MODEL.WEIGHTS == "":
        ## backbone initialization
        backbone_pretrained = backbone_cfg.get("PRETRAINED", "")

        if backbone_pretrained == "":
            logger.warning("Randomly initialize weights for backbone!")
        elif backbone_pretrained in ["timm", "internal"]:
            # skip if it has already been initialized by pretrained=True
            logger.info("Check if the backbone has been initialized with its own method!")
            if backbone_pretrained == "timm":
                if init_backbone_args.pretrained and init_backbone_args.in_chans != 3:
                    load_timm_pretrained(
                        model.backbone, in_chans=init_backbone_args.in_chans, adapt_input_mode="custom", strict=False
                    )
                    logger.warning("override input conv weight adaptation of timm")
        else:
            # initialize backbone with official weights
            tic = time.time()
            logger.info(f"load backbone weights from: {backbone_pretrained}")
            load_checkpoint(model.backbone, backbone_pretrained, strict=False, logger=logger)
            logger.info(f"load backbone weights took: {time.time() - tic}s")

    model.to(torch.device(cfg.MODEL.DEVICE))
    return model, optimizer