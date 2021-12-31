#!/usr/bin/env python3

from det3d.torchie.parallel import MegDataParallel
from det3d.torchie.parallel import collate_kitti
from det3d.models import build_detector
from demo_utils import get_dataset, infer_model
from det3d import torchie
import numpy as np
import time
import os



# DATA_DIR = "/shared_area/kitti_data/velo_28"
DATA_DIR = "/shared_area/datasets/velodyne"
CONFIG_FILE = "/workspace/ros_ws/src/se_ssd/assets/config.py"
CHECKPOINT = "/workspace/ros_ws/src/se_ssd/assets/se-ssd-model.pth"

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


def infer_pc_file(pc_file):
    pc_arr = np.fromfile(pc_file, dtype="float32").reshape(-1, 4)
    pc_arr[:, 3] /= pc_arr[:, 3].max()
    tick = time.perf_counter()
    dataset = get_dataset(pc_arr, cfg.data.test)
    output = infer_model(dataset, model)
    tock = time.perf_counter()
    if predictions is not None and len(predictions) > 0:
        bboxes3d = predictions[0]["box3d_lidar"].cpu()    # batch 1st sample batch_size=1
        bboxes3d[:, -1] *= -1     # batch 1st sample batch_size=1
        print("\n", "--- "*11)
        print(f"---> [inference-time]: \t {(tock-tick)*1000:0.0f} (ms)")
        print(f"---> [sample-prediction-size]: \t {bboxes3d.size()}")
#        print(f"---> [sample-prediction-size]: \n {bboxes3d}")
        np.savez("/shared_area/predictions/"+os.path.splitext(os.path.basename(pc_file))[0]+".npz", boxes=bboxes3d)

if __name__ == "__main__":
    pc_names_list = sorted(os.listdir(DATA_DIR))
    data_size = len(pc_names_list)
    for idx, pc_file in enumerate(pc_names_list):
        infer_pc_file(DATA_DIR+"/"+pc_file)
        print(f"[infered] --- ({idx}/{data_size})")
        t1 = time.perf_counter()
        dataset = get_dataset(pc_arr, cfg.data.test)
        output = infer_model(dataset, model)
        t2 = time.perf_counter()
        print(f"[runtime]: {(t2-t1)*1000:.3f}(ms)")