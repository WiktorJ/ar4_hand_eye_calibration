#!/usr/bin/env python3
"""
A script to follow an aruco marker with a robot arm using PyMoveit2.
"""
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time

import tf2_ros
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from ros2_aruco_interfaces.msg import ArucoMarkers
from tf2_geometry_msgs import do_transform_pose
from pymoveit2 import MoveIt2


class ArucoMarkerFollower(Node):

    def __init__(self):
        super().__init__("aruco_marker_follower")
        self.logger = self.get_logger()

        self.arm_joint_names = [
            "camerajoint_1", "camerajoint_2", "camerajoint_3", "camerajoint_4",
            "camerajoint_5", "camerajoint_6"
        ]
        self.moveit2 = None
        self.init_moveit_timer = self.create_timer(3.0,
                                                   self.initialize_moveit)

        # ID of the aruco marker mounted on the robot
        self.marker_id = self.declare_parameter(
            "marker_id", 1).get_parameter_value().integer_value

        self.subscription = self.create_subscription(ArucoMarkers,
                                                     "/aruco_markers",
                                                     self.handle_aruco_markers,
                                                     1)
        self.pose_pub = self.create_publisher(PoseStamped, "/cal_marker_pose",
                                              1)

        self.target_pose_pub = self.create_publisher(
            PoseStamped, "/follow_aruco_target_pose", 1)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._prev_marker_pose = None

    def initialize_moveit(self):
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=self.arm_joint_names,
            base_link_name="camerabase_link",
            end_effector_name="cameralink_6",
            group_name="ar_manipulator",
            callback_group=ReentrantCallbackGroup(),
        )
        self.moveit2.planner_id = "RRTConnectkConfigDefault"
        self.moveit2.max_velocity = 1.0
        self.moveit2.max_acceleration = 1.0
        self.init_moveit_timer.cancel()

    def handle_aruco_markers(self, msg: ArucoMarkers):
        try:
            self.logger.info(f"Got new aruco marker: {msg}")
            cal_marker_pose = None
            for i, marker_id in enumerate(msg.marker_ids):
                if marker_id == self.marker_id:
                    cal_marker_pose = msg.poses[i]
                    break
                else:
                    self.logger.info(
                        f"Detected unexpected marker with ID: {marker_id}")

            if cal_marker_pose is None:
                return

            # only start following if the marker pose has changed by at least 2cm
            if self._prev_marker_pose is not None:
                if ((cal_marker_pose.position.x -
                     self._prev_marker_pose.position.x) ** 2 +
                        (cal_marker_pose.position.y -
                         self._prev_marker_pose.position.y) ** 2 +
                        (cal_marker_pose.position.z -
                         self._prev_marker_pose.position.z) ** 2 > 0.02 ** 2):
                    self._prev_marker_pose = cal_marker_pose
                    return

            self._prev_marker_pose = cal_marker_pose

            # get pose in robot base frame
            try:
                transformed_pose = self._transform_pose(cal_marker_pose,
                                                        "camera_color_optical_frame",
                                                        "camerabase_link")
                self.logger.info(
                    f"Following marker at pose: {transformed_pose}")
                self.move_to(transformed_pose)
            except Exception as e:
                self.logger.error(f"Error transforming pose or move_to: {e}")
                return

        except Exception as e:
            self.logger.error(f"Error in handle_aruco_markers: {e}")

    def _transform_pose(self, pose: Pose, source_frame,
                        target_frame: str) -> Pose:
        # Get the transform from source frame to target frame
        transform = self.tf_buffer.lookup_transform(target_frame, source_frame,
                                                    Time())
        # Transform the pose
        transformed_pose = do_transform_pose(pose, transform)
        # publish pose
        stamped_pose = PoseStamped()
        stamped_pose.header.frame_id = target_frame
        stamped_pose.pose = transformed_pose
        self.pose_pub.publish(stamped_pose)

        pose.position.z += 0.05
        transformed_pose = do_transform_pose(pose, transform)

        stamped_pose = PoseStamped()
        stamped_pose.header.frame_id = target_frame
        stamped_pose.pose = transformed_pose
        self.target_pose_pub.publish(stamped_pose)
        return transformed_pose

    def move_to(self, msg: Pose):
        if self.moveit2 is None:
            self.logger.error("Moveit2 is not initialized")
            return
        try:
            pose_goal = PoseStamped()
            pose_goal.header.frame_id = "camerabase_link"
            pose_goal.pose = msg

            result = self.moveit2.move_to_pose(pose=pose_goal)
            if result:
                self.moveit2.wait_until_executed()
                self.logger.info("Finished moving with moveit2")
            else:
                self.logger.warn("Failed to move with moveit2")
        except Exception as e:
            self.logger.error(f"Exception in move_to: {e}")


def main():
    rclpy.init()
    node = ArucoMarkerFollower()
    executor = MultiThreadedExecutor(4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
