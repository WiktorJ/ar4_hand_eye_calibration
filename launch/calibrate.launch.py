import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

# This is hard-coded because .rviz configs are not configurable and also have this prefix hard-coded
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
            'rs_compat': 'true',
            'name': _TF_PREFIX
        }.items()
    )

    ar_moveit_launch = PythonLaunchDescriptionSource(
        [
            os.path.join(
                get_package_share_directory("annin_ar4_moveit_config"),
                "launch",
                "moveit.launch.py",
            )
        ]
    )
    rviz_config_file = os.path.join(
        get_package_share_directory("ar4_hand_eye_calibration"),
        "rviz",
        "moveit_with_camera.rviz",
    )
    ar_moveit_args = {
        "include_gripper": "False",
        "rviz_config_file": rviz_config_file,
    }.items()
    ar_moveit = IncludeLaunchDescription(
        ar_moveit_launch, launch_arguments=ar_moveit_args
    )

    aruco_params = os.path.join(
        get_package_share_directory("ar4_hand_eye_calibration"),
        "config",
        "aruco_parameters.yaml",
    )
    aruco_recognition_node = Node(
        package="ros2_aruco", executable="aruco_node", parameters=[aruco_params]
    )

    calibration_args = {
        "name": "ar4_calibration",
        "calibration_type": "eye_on_base",
        "robot_base_frame": f"{_TF_PREFIX}base_link",
        "robot_effector_frame": f"{_TF_PREFIX}ee_link",
        "tracking_base_frame": "camera_color_optical_frame",
        "tracking_marker_frame": "calibration_aruco",
    }

    calibration_aruco_publisher = Node(
        package="ar4_hand_eye_calibration",
        executable="calibration_aruco_publisher.py",
        name="calibration_aruco_publisher",
        output="screen",
        parameters=[
            {
                "tracking_base_frame": calibration_args["tracking_base_frame"],
                "tracking_marker_frame": calibration_args[
                    "tracking_marker_frame"],
                "marker_id": 1,
            }
        ],
    )

    easy_handeye2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("easy_handeye2"),
                    "launch",
                    "calibrate.launch.py",
                )
            ]
        ),
        launch_arguments=calibration_args.items(),
    )

    # static transform publisher for camera_link to world
    static_tf_publisher = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0", "0", "0", "0", "0", "0", "world", "camera_link"],
        output="screen",

    )

    ld = LaunchDescription()
    ld.add_action(depthai_camera)
    ld.add_action(static_tf_publisher)
    ld.add_action(ar_moveit)
    ld.add_action(aruco_recognition_node)
    ld.add_action(calibration_aruco_publisher)
    ld.add_action(easy_handeye2)
    return ld

