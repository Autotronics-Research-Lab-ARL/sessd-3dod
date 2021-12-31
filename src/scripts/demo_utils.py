
#!/usr/bin/env python3

from det3d.torchie.trainer.trainer import example_to_device
from det3d.torchie.parallel import collate_kitti
from det3d.datasets import build_dataset
from det3d.utils.dist import dist_common
from det3d import torchie
from visualization_msgs.msg import Marker
from arl_msgs.msg import BBox3D
from torch.utils.data import DataLoader
from collections import defaultdict
from typing import List
import pyquaternion
import numpy as np
import torch
import rospy
# import tf


def logit(func):
    def wrapper(*args, **kwargs):
        tick = time.perf_counter()
        result = func(*args, **kwargs)
        tock = time.perf_counter()
        model_runtime = tock - tick    # inference time in seconds
        print("\n", "--- "*11)
        print(f"---> [inference-time]: \t {model_runtime*1000:0.0f} (ms)")
        print(f"---> [sample-prediction-size]: \t {result[0].size()}")
        return result
    return wrapper


def splitHalfPointCloud(pointCloud:np.ndarray)->List[np.ndarray]:
    '''Splits the pointcloud data into head half and tail half'''
    X, Y = pointCloud[:, 0], pointCloud[:, 1]
    Azimuth = np.arctan2(Y, X)
    headMask = np.logical_and(Azimuth >= np.deg2rad(-90), Azimuth <= np.deg2rad(90))
    secondQuadMask = np.logical_and(Azimuth >= np.deg2rad(90), Azimuth <= np.deg2rad(180))
    thirdQuadMask = np.logical_and(Azimuth >= np.deg2rad(-180), Azimuth <= np.deg2rad(-90))
    tailMask = np.logical_or(secondQuadMask, thirdQuadMask)
    return [pointCloud[headMask], pointCloud[tailMask]]


def get_dataset(pointcloud:np.ndarray, cfg:dict):
    '''Builds and Loads Dataset to feed the model'''
    dataset = build_dataset(cfg)
    headFrame, tailFrame = splitHalfPointCloud(pointcloud)
    tailFrame = rotatePointCloudAroundZ(tailFrame, 180)
    dataset.pointcloud_batch = [headFrame, tailFrame]
    dataloader = DataLoader(
        [*dataset], batch_size=2, 
        collate_fn=collate_kitti, shuffle=False
    ) # torch.DataLoader
    dataset = dataloader.dataset         # det3d.datasets.kitti.kitti.KittiDataset
    return dataset


def infer_model(dataset, model, device="cuda"):
    """
    Predicts the 3D-Boxes <locations, heading, class-ids, scores>

    Params
    ------
    dataset: (DemoDataset)
        loads the dataset with the preprocessing pipeline and configs

    model: (torch.Module)
       the model to be infered

    device: (str)
        the device to be used to run the model

    """
    device = torch.device(device)        # device(type='cuda')
    num_devices = dist_common.get_world_size()       # 1
    batch_samples = collate_kitti(dataset, samples_per_gpu=2)
    example = example_to_device(batch_samples, device=torch.device(device))
    with torch.no_grad():
        # outputs: predicted results in lidar coord.
        predictions = model(example, return_loss=False, rescale=True)
        outputs  = mergeSplittedPredictions(predictions)
        return outputs


def mergeSplittedPredictions(pred_dict)->dict:
    result_dict = {}
    head_dict, tail_dict = pred_dict
    if len(tail_dict['box3d_lidar']) > 0:
        tail_dict['box3d_lidar'] = tail_dict['box3d_lidar'].cpu().numpy()
        tail_dict['box3d_lidar'][:, :3] = rotatePointCloudAroundZ(tail_dict['box3d_lidar'][:, :3], -180)
    result_dict["boxes3d"] = np.concatenate([head_dict['box3d_lidar'].cpu().numpy(), tail_dict['box3d_lidar']])
    result_dict['scores'] = torch.cat([head_dict['scores'], tail_dict['scores']]).cpu().numpy()
    result_dict['labels'] = torch.cat([head_dict['label_preds'], tail_dict['label_preds']]).cpu().numpy()
    return result_dict


def rotatePointCloudAroundZ(points:np.ndarray, angle:int)->np.ndarray:
    '''Rotates pointcloud data around the z-axis'''
    angle = np.deg2rad(angle)
    rotationMtx = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    rotated_pts = np.float32(rotationMtx @ points[:, :3].T)
    return np.hstack([rotated_pts.T, points[:, 3:4]])


def numpy_to_MarkerMsg(box_arr, name_space, idx, lifetime, frame_id=''):
    """
    Converts Numpy bounding 3d-box array to BBox3D Msg

    Params
    ------
    box_arr : (np.1d_array)
        bounding box of shape (7,) box<center, size, heading>

    name_space: (str)
        namespace that groups boxes with the same namespace

    idx: (int)
        unique box id for distinction

    lifetime: (int), (float)
        life time of the marker msg in seconds

    frame_id: (str)
        the frame origin the marker will be relative to


    Returns
    -------
    bb3D : (BBox3D)
        bounding box of shape (8,) box<center, size, heading, class>


    """
    msg = Marker()
    (x, y, z), (dx, dy, dz) = box_arr[:3], box_arr[3:6]
    # (qx, qy, qz, qw) = tf.transformations.quaternion_from_euler(0, 0, box_arr[6])
    (qw, qx, qy, qz) = pyquaternion.Quaternion(axis=[0, 0, 1], angle=-box_arr[6]).normalised.q
    # -------------------------------------------
    msg.header.frame_id = frame_id
    msg.header.stamp = rospy.Time.now()
    msg.ns, msg.id = name_space, idx
    msg.type = Marker.CUBE
    msg.action = Marker.ADD
    msg.lifetime = rospy.Duration(lifetime)
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    msg.pose.orientation.x = qx
    msg.pose.orientation.y = qy
    msg.pose.orientation.z = qz
    msg.pose.orientation.w = qw
    msg.scale.x = dx
    msg.scale.y = dy
    msg.scale.z = dz
    msg.color.r = 0.
    msg.color.g = 1.0
    msg.color.b = 0.
    msg.color.a = 0.50
    return msg


def numpy_to_BBox3DMsg(np_box, class_id, score):
    """
    Converts Numpy bounding 3d-box array to BBox3D Msg

    Params
    ------
    np_box : (np.float32)
        bounding box of shape (7,) box<center, size, heading>

    class_id: (int)
        bounding box class-id

    score: (float)
        bounding box confidence score


    Returns
    -------
    msg : (BBox3D)
        detection/BBox3D custom ROS message for 3D bounding boxes
    """
    msg = BBox3D()
    msg.center.x, msg.center.y, msg.center.z = np_box[:3]
    msg.size.x, msg.size.y, msg.size.z = np_box[3:6]
    msg.heading = np_box[6]
    msg.class_id = int(class_id)
    msg.score = score
    return msg


def BBox3DMsg_to_numpy(bbox3D_msg):
    """
    Converts BBox3D Msg to Numpy array

    Params
    ------
    bbox3D_msg : (BBox3D)
        detection/BBox3D custom ROS message for 3D bounding boxes


    Returns
    -------
    bb3D : (BBox3D)
        bounding box of shape (8,) box<center, size, heading, class>

    """
    bbox_arr = []
    bbox_arr.append(bbox3D_msg.center.x)
    bbox_arr.append(bbox3D_msg.center.y)
    bbox_arr.append(bbox3D_msg.center.z)
    bbox_arr.append(bbox3D_msg.size.x)
    bbox_arr.append(bbox3D_msg.size.y)
    bbox_arr.append(bbox3D_msg.size.z)
    bbox_arr.append(bbox3D_msg.heading)
    bbox_arr.append(bbox3D_msg.class_id)
    return np.float32(bbox_arr)
