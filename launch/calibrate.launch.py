import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
import launch.substitutions
from launch.launch_description_sources import PythonLaunchDescriptionSource

# This is hard-coded because .rviz configs are not configurable and also have this prefix hard-coded
_TF_PREFIX = 'camera'

def generate_launch_description():
    current_pkg_dir = get_package_share_directory("ar4_hand_eye_calibration")

    # Declare launch arguments for the orchestrator
    declared_args = []
    declared_args.append(
        DeclareLaunchArgument(
            "joint_states_yaml_path",
            default_value=os.path.join(current_pkg_dir, "config", "calibration_joint_states.yaml"),
            description="Path to the YAML file containing robot joint states for calibration.",
        )
    )
    # Add other orchestrator related arguments if needed, e.g., thresholds, move_group_name
    # For now, we'll use defaults in the orchestrator node or pass them via params file if preferred.
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
        current_pkg_dir,
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

    # aruco_params = os.path.join(
    #     current_pkg_dir,
    #     "config",
    #     "aruco_parameters.yaml",
    # )
    # aruco_recognition_node = Node(
    #     package="ros2_aruco", executable="aruco_node", parameters=[aruco_params]
    # )
    aruco_params = os.path.join(
        current_pkg_dir,
        "config",
        "aruco_board_parameters.yaml",
    )
    aruco_recognition_node = Node(
        package="ros2_aruco", executable="aruco_board_node", parameters=[aruco_params]
    )

    handeye_calibration_name = "ar4_calibration" # Consistent name
    calibration_args = {
        "name": handeye_calibration_name,
        "calibration_type": "eye_on_base",
        "robot_base_frame": f"{_TF_PREFIX}base_link", # Frame MoveIt uses as planning frame for the arm
        "robot_effector_frame": f"{_TF_PREFIX}ee_link", # Frame MoveIt uses as end-effector for the arm
        "tracking_base_frame": f"{_TF_PREFIX}_color_optical_frame", # Optical frame of the camera
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

    # Calibration Orchestrator Node
    calibration_orchestrator_node = Node(
        package="ar4_hand_eye_calibration",
        executable="calibration_orchestrator.py",
        name="calibration_orchestrator",
        output="screen",
        parameters=[
            {
                "joint_states_yaml_path": launch.substitutions.LaunchConfiguration("joint_states_yaml_path"),
                "move_group_name": "ar_manipulator", # Make this a launch arg if it can change
                # "handeye_calibration_name": handeye_calibration_name,
                # "min_translation_threshold_m": 0.02, # Example override
                # "min_rotation_threshold_deg": 5.0,   # Example override
                "num_joints": 6 # Set according to your robot (AR4 is 6DOF)
            }
        ],
    )

    # static transform publisher for camera_link to world
    static_tf_publisher = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0", "0", "0", "0", "0", "0", "world", "camera_link"],
        output="screen",

    )

    ld = LaunchDescription()
    for arg in declared_args:
        ld.add_action(arg)

    ld.add_action(depthai_camera)
    ld.add_action(static_tf_publisher)
    ld.add_action(ar_moveit)
    ld.add_action(aruco_recognition_node)
    ld.add_action(calibration_aruco_publisher)
    ld.add_action(easy_handeye2)
    ld.add_action(calibration_orchestrator_node) # Add the orchestrator
    return ld

