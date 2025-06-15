import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


_TF_PREFIX = 'camera'
def generate_launch_description():
    depthai_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("depthai_ros_driver"),
                    "launch",
                    "camera.launch.py",
                )
            ]
        ),
        launch_arguments={
            'rs_compat': "true",
            'name': _TF_PREFIX
        }.items()
    )

    return LaunchDescription(
        [
            depthai_camera,
        ]
    )
