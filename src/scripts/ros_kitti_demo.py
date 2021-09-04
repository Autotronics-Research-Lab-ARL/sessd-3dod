#!/usr/bin/env python3

from det3d.torchie.parallel import MegDataParallel
from det3d.torchie.parallel import collate_kitti
from det3d.datasets import build_dataset
from det3d.datasets import build_dataset
from torch.utils.data import DataLoader
from det3d.models import build_detector
from det3d import torchie
from demo_utils import numpy_to_MarkerMsg
from demo_utils import numpy_to_BBox3DMsg
from demo_utils import infer_model
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray
from arl_msgs.msg import BBox3DArray
import numpy as np
import ros_numpy
import rospy
import time



CONFIG_FILE = "/workspace/src/se_ssd/assets/config.py"
CHECKPOINT = "/workspace/src/se_ssd/assets/se-ssd-model.pth"

cfg = torchie.Config.fromfile(CONFIG_FILE)
# cfg.data.val.test_mode = True

model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

# checkpoint_path = os.path.join(cfg.work_dir, args.checkpoint)
checkpoint = torchie.trainer.load_checkpoint(model, CHECKPOINT, map_location="cpu")
# print("--------> checkpoint type: ", type(checkpoint))

if "CLASSES" in checkpoint["meta"]: model.CLASSES = checkpoint["meta"]["CLASSES"]
else: model.CLASSES = dataset.CLASSES

model = MegDataParallel(model, device_ids=[0])
model.eval()


def pointcloud_handler(pc_msg):
    global marker_pub, bboxes_pub
    pc_struct = ros_numpy.numpify(pc_msg)
    pc_arr = np.array([
        pc_struct['x'],
        pc_struct['y'],
        pc_struct['z'],
        pc_struct['i'],
    ], dtype='float32').T
    dataset = build_dataset(cfg.data.test)
    dataset.pc_arr = pc_arr
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=None,
        num_workers=8,
        collate_fn=collate_kitti,
        shuffle=False,
    )
    tick = time.perf_counter()
    predictions = infer_model(data_loader, model)
    tock = time.perf_counter()
    model_runtime = tock - tick    # inference time in seconds
    if predictions is not None and len(predictions) > 0:
        bboxes3d = predictions[0]["box3d_lidar"]    # batch 1st sample batch_size=1
        class_ids = predictions[0]["label_preds"]
        scores = predictions[0]["scores"]
        print("\n", "--- "*11)
        print(f"---> [inference-time]: \t {model_runtime*1000:0.0f} (ms)")
        print(f"---> [sample-prediction-size]: \t {bboxes3d.size()}")
        # print(f"---> [sample-prediction-size]: \n {predictions[0]}")
        markers_msg = MarkerArray()
        bboxes_msg = BBox3DArray()
        for idx in range(len(bboxes3d)):
            marker = numpy_to_MarkerMsg(bboxes3d[idx], "bbox3d", idx, model_runtime*1.3, lidar_frame)
            box_msg = numpy_to_BBox3DMsg(bboxes3d[idx], class_ids[idx], scores[idx])
            markers_msg.markers.append(marker)
            bboxes_msg.boxes.append(box_msg)
        vis_pc_msg = ros_numpy.msgify(PointCloud2, pc_struct, frame_id=lidar_frame)
        marker_pub.publish(markers_msg)
        bboxes_pub.publish(bboxes_msg)
        vis_pc_pub.publish(vis_pc_msg)


if __name__ == "__main__":
    # init ros-node
    rospy.init_node("object_detector_3d_node")

    # topic names
    pointcloud_topic = rospy.get_param("pointcloud_topic")
    boxes3d_lidar_topic = rospy.get_param("boxes3d_lidar_topic")
    rviz_boxes_marker_topic = rospy.get_param("rviz_boxes_marker_topic")
    rviz_pointcloud_topic = rospy.get_param("rviz_pointcloud_topic")
    rviz_pc_queue_size =  rospy.get_param("rviz_pc_queue_size")
    boxes3d_queue_size = rospy.get_param("boxes3d_queue_size")
    markers_queue_size = rospy.get_param("markers_queue_size")
    lidar_frame = rospy.get_param("lidar_frame")

    # subscribers
    pc_sub = rospy.Subscriber(pointcloud_topic, PointCloud2, pointcloud_handler)

    # publishers
    vis_pc_pub = rospy.Publisher(rviz_pointcloud_topic, PointCloud2, queue_size=rviz_pc_queue_size)
    bboxes_pub = rospy.Publisher(boxes3d_lidar_topic, BBox3DArray, queue_size=boxes3d_queue_size)
    marker_pub = rospy.Publisher(rviz_boxes_marker_topic, MarkerArray, queue_size=markers_queue_size)

    rospy.spin()
