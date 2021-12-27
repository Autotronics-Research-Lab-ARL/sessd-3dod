from det3d.datasets.custom import PointCloudDataset
from det3d.datasets.registry import DATASETS
from det3d.core.bbox import box_np_ops
import numpy as np



image_shape = (375, 1242)
calib_dict = {
    "rect": np.array(
      [[ 0.9999239 ,  0.00983776, -0.00744505,  0.        ],
       [-0.0098698 ,  0.9999421 , -0.00427846,  0.        ],
       [ 0.00740253,  0.00435161,  0.9999631 ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]
    ),
    "Trv2c": np.array(
      [[ 2.34773698e-04, -9.99944155e-01, -1.05634778e-02, -2.79681694e-03],
       [ 1.04494074e-02,  1.05653536e-02, -9.99889574e-01, -7.51087914e-02],
       [ 9.99945389e-01,  1.24365378e-04,  1.04513030e-02, -2.72132796e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
    ),
    "P2": np.array(
      [[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01],
       [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
#      [[1, 0.000000e+00, 6.095593e+02, 4.485728e+01],
#       [0.000000e+00, 1, 1.728540e+02, 2.163791e-01],
       [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]]
    ) 
}

calib = {
    **calib_dict,
    "frustum": box_np_ops.get_valid_frustum(
        calib_dict["rect"],
        calib_dict["Trv2c"],
        calib_dict["P2"],
        image_shape
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
            str(),	# info_path
            cfg=cfg,
            test_mode=True,
            pipeline=pipeline,
            class_names="Car",
            **kwrags
        )
        self.pc_arr = None
        self.calib = calib

    def __len__(self):
        return 1

    def _get_sensor_data(self):
        sensor_data = {
            "type": "DemoDataset",
            "lidar": {
                "type": "lidar",
                "points": self.pc_arr,
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
                "image_shape": image_shape,
            },
        }
        return sensor_data

    def __getitem__(self, index):
        sensor_data = self._get_sensor_data()
        data, _ = self.pipeline(sensor_data, None)
        return data

