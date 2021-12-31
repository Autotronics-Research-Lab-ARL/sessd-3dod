import itertools
import logging
from pathlib import Path
from det3d.builder import build_box_coder
from det3d.utils.config_tool import get_downsample_factor


dataset_type = "DemoDataset"
tasks = [dict(num_class=1, class_names=["Car"]),]


class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))
box_coder = dict(type="ground_box3d_coder", n_dim=7, linear_dim=False, encode_angle_vector=False,)


my_paras = dict(
    batch_size=2,
    data_mode="train",        # "train" or "trainval": the set to train the model;
    enable_ssl=True,         # Ensure "False" in CIA-SSD training
    eval_training_set=False,  # True: eval on "data_mode" set; False: eval on validation set.[Ensure "False" in training; Switch in Testing]

    # unused
    enable_difficulty_level=False,
    remove_difficulty_points=False,  # act with neccessary condition: enable_difficulty_level=True.
    far_points_first=False,
    data_aug_with_context=-1,        # enlarged size for w and l in data aug.
    loss_iou=None,
)


# model settings
norm_cfg = None
model = dict(
    type="VoxelNet",
    pretrained=None,
    reader=dict(type="VoxelFeatureExtractorV3", num_input_features=4, norm_cfg=norm_cfg,),
    backbone=dict(type="SpMiddleFHD", num_input_features=4, ds_factor=8, norm_cfg=norm_cfg,),
    neck=dict(
        type="SSFA",
        layer_nums=[5,],
        ds_layer_strides=[1,],
        ds_num_filters=[128,],
        us_layer_strides=[1,],
        us_num_filters=[128,],
        num_input_features=128,
        norm_cfg=norm_cfg,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="MultiGroupHead",
        mode="3d",
        in_channels=sum([128,]),
        norm_cfg=norm_cfg,
        tasks=tasks,
        weights=[1,],
        box_coder=build_box_coder(box_coder),
        encode_background_as_zeros=True,
        loss_norm=dict(type="NormByNumPositives", pos_cls_weight=1.0, neg_cls_weight=1.0,),
        loss_cls=dict(type="SigmoidFocalLoss", alpha=0.25, gamma=2.0, loss_weight=1.0,),
        use_sigmoid_score=True,
        loss_bbox=dict(type="WeightedSmoothL1Loss", sigma=3.0, code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], codewise=True, loss_weight=2.0, ),
        encode_rad_error_by_sin=True,
        loss_aux=dict(type="WeightedSoftmaxClassificationLoss", name="direction_classifier", loss_weight=0.2,),
        direction_offset=0.0,
        #loss_iou=my_paras['loss_iou'],
    ),
)


target_assigner = dict(
    type="iou",
    anchor_generators=[
        dict(
            type="anchor_generator_range",
            sizes=[1.6, 3.9, 1.56],  # w, l, h
            anchor_ranges=[0, -40.0, -1.0, 70.4, 40.0, -1.0],
            rotations=[0, 1.57],
            matched_threshold=0.6, # 0.6
            unmatched_threshold=0.45,
            class_name="Car",
        ),
    ],
    sample_positive_fraction=-1,
    sample_size=512,
    region_similarity_calculator=dict(type="nearest_iou_similarity",),
    pos_area_threshold=-1,
    tasks=tasks,
)


assigner = dict(
    box_coder=box_coder,
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    debug=False,
    enable_similar_type=True,
)


train_cfg = dict(assigner=assigner)

test_cfg = dict(
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=100,
        nms_iou_threshold=0.01,
    ),
    score_threshold=0.3,
    post_center_limit_range=[0, -40.0, -5.0, 70.4, 40.0, 5.0],
    max_per_img=100,
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
    remove_environment=False,
    remove_unknown_examples=False,
)

voxel_generator = dict(
    range=[0, -40.0, -3.0, 70.4, 40.0, 1.0],
    voxel_size=[0.05, 0.05, 0.1],
    max_points_in_voxel=5,
    max_voxel_num=20000,
    far_points_first=my_paras['far_points_first'],
)

test_pipeline = [
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignTarget", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]


data = dict(
    samples_per_gpu=my_paras['batch_size'],  # batch_size: 1
    workers_per_gpu=2,  # default: 2
    val=dict(
        type=dataset_type,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    trainval=dict(
        type=dataset_type,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)

optimizer = dict(type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(type="one_cycle", lr_max=0.003, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,)  # learning policy in training hooks


checkpoint_config = dict(interval=1)
log_config = dict(interval=10,hooks=[dict(type="TextLoggerHook"),],) # dict(type='TensorboardLoggerHook')
