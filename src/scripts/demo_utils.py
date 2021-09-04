
#!/usr/bin/env python3

from det3d.torchie.trainer.trainer import example_to_device
from visualization_msgs.msg import Marker
from arl_msgs.msg import BBox3D
from det3d.torchie.parallel import collate_kitti
from det3d.utils.dist import dist_common
from det3d import torchie
import pyquaternion
import torch
import rospy
# import tf



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



def infer_model(dataloader, model, device="cuda"):
    """
    Predicts the 3D-Boxes <locations, heading, class-ids, scores>

    Params
    ------
    dataloader: (torch.DataLoader)
        loads the dataset with the preprocessing pipeline and configs

    model: (torch.Module)
       the model to be infered

    device: (str)
        the device to be used to run the model

    """
    dataset = dataloader.dataset         # det3d.datasets.kitti.kitti.KittiDataset
    device = torch.device(device)        # device(type='cuda')
    num_devices = dist_common.get_world_size()       # 1
    samples = [dataset[0]]
    batch_samples = collate_kitti(samples)
    example = example_to_device(batch_samples, device=torch.device(device))
    results_dict = {}
    with torch.no_grad():
        # outputs: predicted results in lidar coord.
        outputs = model(example, return_loss=False, rescale=True)
        return outputs

