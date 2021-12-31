from det3d.datasets.custom import PointCloudDataset
from det3d.datasets.registry import DATASETS
from det3d.core.bbox import box_np_ops
import numpy as np



BATCH_SIZE = 2
IMG_SHAPE = (375, 1242)
CALIB_DICT = {
    "rect": np.array([
        [ 1.   ,  0.01 , -0.007,  0.   ],
        [-0.01 ,  1.   , -0.004,  0.   ],
        [ 0.007,  0.004,  1.   ,  0.   ],
        [ 0.   ,  0.   ,  0.   ,  1.   ]
    ]),
    "Trv2c": np.array([
        [ 0.   , -1.   , -0.011, -0.003],
        [ 0.01 ,  0.011, -1.   , -0.075],
        [ 1.   ,  0.   ,  0.01 , -0.272],
        [ 0.   ,  0.   ,  0.   ,  1.   ]
    ]),
    "P2": np.array([
        # [721.538,   0.   , 609.559,  44.857],
        # [  0.   , 721.538, 172.854,   0.216],
        [  1.   ,   0.   , 609.559,  44.857],
        [  0.   ,   1.   , 172.854,   0.216],
        [  0.   ,   0.   ,   1.   ,   0.003]
    ])
}
CAM_CALIB = {
   **CALIB_DICT,
    "frustum": box_np_ops.get_valid_frustum(
        CALIB_DICT["rect"],
        CALIB_DICT["Trv2c"],
        CALIB_DICT["P2"],
        IMG_SHAPE
    )
}

@DATASETS.register_module
class DemoDataset(PointCloudDataset):
    NumPointFeatures = 4
    def __init__(
        self,
        cfg=None,
        pipeline=None,
        class_names=None,
        **kwrags
    ):
        super(DemoDataset, self).__init__(
            str(),       # root_dir
            str(),	     # info_path
            cfg=cfg,
            test_mode=True,
            pipeline=pipeline,
            class_names="Car",
            **kwrags
        )
        self.pointcloud_batch = []
        self.calib = CAM_CALIB
        self.batch_size = BATCH_SIZE

    def __len__(self):
        return BATCH_SIZE

    def _get_sensor_data(self, index):
        sensor_data = {
            "type": "DemoDataset",
            "lidar": {
                "type": "lidar",
                "points": self.pointcloud_batch[index],
                "ground_plane": None,
                "names": None,
                "annotations": None,
                "targets": None
            },
            "labeled": False,
            "calib": self.calib,
            "cam": {
                "annotations": None
            },
            "mode": "val",
            "metadata": {
                "num_point_features": DemoDataset.NumPointFeatures,
                "image_shape": IMG_SHAPE,
            },
        }
        return sensor_data    

    def __getitem__(self, index):
        sensor_data = self._get_sensor_data(index)
        data, _ = self.pipeline(sensor_data, None)
        return data
