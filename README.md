# EPRO-GDR

This repo includes the code to our paper "Probabilistic 6D Pose Estimation for XR Applications" and is based on these repos: [GDRNPP](https://github.com/shanice-l/gdrnpp_bop2022) and [EPro-PnP-v2](https://github.com/tjiiv-cprg/EPro-PnP-v2/tree/main/EPro-PnP-6DoF_v2).

## Path Setting

* `gdrnpp/datasets` (adapted from GDRNPP)

Download the 6D pose datasets from the
[BOP website](https://bop.felk.cvut.cz/datasets/) and
[VOC 2012](https://pjreddie.com/projects/pascal-voc-dataset-mirror/)
for background images.
Please also download the  `test_bboxes` (in the paper we used the real+pbr bounding boxes for T-Less and YCB-V and pbr for LM-O and ITODD) from
here [OneDrive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/liuxy21_mails_tsinghua_edu_cn/Eq_2aCC0RfhNisW8ZezYtIoBGfJiRIZnFxbITuQrJ56DjA?e=hPbJz2) (password: groupji) or [BaiDuYunPan](https://pan.baidu.com/s/1FzTO4Emfu-DxYkNG40EDKw)(password: vp58).

The structure of `datasets` folder should look like below:
```
datasets/
├── BOP_DATASETS   # https://bop.felk.cvut.cz/datasets/
    ├──lmo
    ├──ycbv
    └──tless
└──VOCdevkit
```

Please download the **Base archive**, **Object models**, **PBR-BlenderProc4BOP training images** and **BOP test images** follow BOP's instruction and please note that the datasets LM and LM-O share the same train_pbr.

For example, the folder `lmo` should look like:
```
lmo
├── camera.json
├── dataset_info.md
├── models
├── models_eval
├── test
├── test_targets_bop19.json
└── train_pbr
```

## Requirements

```
docker pull jili233/gdrnpp:v14
```
* * Run this container with `-it` and the directory of this repo should be mounted onto the container's `/home` directory.

* Conda will be activated by default to compile the extensions and run our method, we need to deactivate conda by running 2x (or 3x) `conda deactivate`.

* Compile the extensions of GDRNPP (only need to compile fps and egl_renderer).

```
sh scripts/compile_all.sh
```

## Training of Our Method
Please download the trained models at [Onedrive](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/liuxy21_mails_tsinghua_edu_cn/EgOQzGZn9A5DlaQhgpTtHBwB2Bwyx8qmvLauiHFcJbnGSw?e=EZ60La) (password: groupji) or [BaiDuYunPan](https://pan.baidu.com/s/1LhXblEic6pYf1i6hOm6Otw)(password: 10t3) and put them in the folder `gdrnpp/output`.

The structure of the training command is as follows:

`./core/gdrn_modeling/train_gdrn.sh <config_path> <gpu_ids> (other args)`

For example, to start the fine-tuning from the checkpoints of GDRNPP:

* LM-O

`./core/gdrn_modeling/train_gdrn.sh configs/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo.py 0`

* YCB-V

`./core/gdrn_modeling/train_gdrn.sh configs/gdrn/ycbv/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ycbv.py 0`

* T-Less

`./core/gdrn_modeling/train_gdrn.sh configs/gdrn/tless/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_tless.py 0`

In order to reduce the wall-clock time we increase the batch size from 48 (in GDRNPP) to 72, thus we need a GPU with more memory for training. Note that every hyperparameter is important for fine-tuning, if trained with smaller batch size, the results might be different from the paper.

The number of workers can be changed by changing this key `DATALOADER.NUM_WORKERS` in the config files under the folder `configs/gdrn`.

During the training, the checkpoints will be saved every 2 epochs and this period can be changed by changing this key `CHECKPOINT_PERIOD` at [here](gdrnpp/configs/_base_/common_base.py).

## Testing of Our Method
Please download the checkpoints of our method at [Onedrive](https://studtudarmstadtde-my.sharepoint.com/:f:/g/personal/jiayin_li_stud_tu-darmstadt_de/EqB1CgjK_4hAmrhllRLMpm0BRiH8GcNyHUUqbDbvJ0KhKQ?e=9M5qmG) and put them under the corresponding folders: `output/gdrn` + `itodd_pbr`, `lmo_pbr`, `tless`, `ycbv`.

The structure of the testing command is as follows:

`./core/gdrn_modeling/test_gdrn_depth_refine.sh <config_path> <gpu_ids> <ckpt_path> (other args)`

For example, to test the best checkpoints provided here ([Onedrive](https://studtudarmstadtde-my.sharepoint.com/:f:/g/personal/jiayin_li_stud_tu-darmstadt_de/EqB1CgjK_4hAmrhllRLMpm0BRiH8GcNyHUUqbDbvJ0KhKQ?e=9M5qmG)):

* LM-O:

`./core/gdrn_modeling/test_gdrn_depth_refine.sh configs/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo.py 0 output/gdrn/lmo_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_lmo/model_wo_optim_lmo.pth`

* YCB-V:

`./core/gdrn_modeling/test_gdrn_depth_refine.sh configs/gdrn/ycbv/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ycbv.py 0 output/gdrn/ycbv/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_ycbv/model_wo_optim_ycbv.pth`

* T-Less:

`./core/gdrn_modeling/test_gdrn_depth_refine.sh configs/gdrn/tless/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_tless.py 0 output/gdrn/tless/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_tless/model_wo_optim_tless.pth`