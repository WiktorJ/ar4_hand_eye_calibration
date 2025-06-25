#!/usr/bin/env python3
"""
A simplified version of the ArUco marker follower that separates marker detection from movement.
"""

import rclpy
from rclpy.node import Node
import threading
import time
from queue import Queue

import tf2_ros
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from ros2_aruco_interfaces.msg import ArucoMarkers
from tf2_geometry_msgs import do_transform_pose
from pymoveit2 import MoveIt2
from rcl_interfaces.msg import ParameterDescriptor, ParameterType


# This is hard-coded because .rviz configs are not configurable and also have this prefix hard-coded
_TF_PREFIX = "camera"


class ArucoBoardFollower(Node):
    def __init__(self):
        super().__init__("follow_aruco_board")
        self.logger = self.get_logger()

        # Create TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Create publishers
        # self.pose_pub = self.create_publisher(PoseStamped, "/cal_marker_pose",
        #                                       10)
        # self.target_pose_pub = self.create_publisher(PoseStamped,
        #                                              "/follow_aruco_target_pose",
        #                                              10)

        # Create subscription to the ArUco markers
        self.subscription = self.create_subscription(
            ArucoMarkers, "/aruco_board_pose", self.handle_aruco_board_pose, 10
        )

        # State variables
        self._prev_marker_pose = None
        self.is_moving = False
        self.pose_queue = Queue()

        # Create a timer for status updates
        self.create_timer(5.0, self.log_status)

        # Create a separate thread for movement
        self.executor_thread = threading.Thread(
            target=self.movement_thread, daemon=True
        )
        self.executor_thread.start()
        self.arm_joint_names = [
            f"{_TF_PREFIX}joint_1",
            f"{_TF_PREFIX}joint_2",
            f"{_TF_PREFIX}joint_3",
            f"{_TF_PREFIX}joint_4",
            f"{_TF_PREFIX}joint_5",
            f"{_TF_PREFIX}joint_6",
        ]
        self.base_link = f"{_TF_PREFIX}base_link"
        self.ee_link = f"{_TF_PREFIX}ee_link"
        self.link_6 = f"{_TF_PREFIX}link_6"
        self.marker_frame = f"aruco_marker_{self.marker_id}"

        self.logger.info("ArucoMarkerFollower node initialized")

    def log_status(self):
        """Log current status"""
        self.logger.info(
            f"Is moving: {self.is_moving}, Queue size: {self.pose_queue.qsize()}"
        )

    def movement_thread(self):
        """Thread for executing robot movements"""
        # Initialize MoveIt2 in this thread
        self.logger.info("Initializing MoveIt2 in movement thread")
        # Create MoveIt2 client
        try:
            moveit2_node = rclpy.create_node("moveit2_client")
            self.moveit2 = MoveIt2(
                node=moveit2_node,
                joint_names=self.arm_joint_names,
                base_link_name=self.base_link,
                end_effector_name=self.link_6,
                group_name="ar_manipulator",
            )
            self.moveit2.planner_id = "RRTConnectkConfigDefault"
            self.moveit2.max_velocity = 1.0
            self.moveit2.max_acceleration = 1.0

            # Create a mini-executor for this node
            moveit_executor = rclpy.executors.SingleThreadedExecutor()
            moveit_executor.add_node(moveit2_node)

            # Spin in a separate thread
            spin_thread = threading.Thread(
                target=lambda: moveit_executor.spin(), daemon=True
            )
            spin_thread.start()

            # Process movement requests
            while rclpy.ok():
                try:
                    # Get a pose from the queue if available
                    if not self.pose_queue.empty() and not self.is_moving:
                        self.is_moving = True
                        pose = self.pose_queue.get(timeout=0.1)

                        # Execute movement
                        self.logger.info(
                            f"Processing movement to {pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}"
                        )

                        pose_goal = PoseStamped()
                        pose_goal.header.frame_id = self.base_link
                        pose_goal.header.stamp = moveit2_node.get_clock().now().to_msg()
                        pose_goal.pose = pose

                        self.moveit2.move_to_pose(pose=pose_goal)
                        self.moveit2.wait_until_executed()
                        self.logger.info("Movement completed")
                        self.is_moving = False
                    else:
                        # Sleep to avoid busy waiting
                        time.sleep(0.1)
                except Exception as e:
                    self.logger.error(f"Error in movement thread: {e}")
                    self.is_moving = False
        except Exception as e:
            self.logger.error(f"Error setting up movement thread: {e}")

    def handle_aruco_board_pose(self, msg):
        """Handle incoming ArUco marker detections"""
        # add more comprehensive inline docs ai!

        cal_marker_pose = msg.pose
        # Check if marker has moved enough
        if self._prev_marker_pose is not None:
            dist_squared = (
                (cal_marker_pose.position.x - self._prev_marker_pose.position.x) ** 2
                + (cal_marker_pose.position.y - self._prev_marker_pose.position.y) ** 2
                + (cal_marker_pose.position.z - self._prev_marker_pose.position.z) ** 2
            )

            if dist_squared < 0.02**2:
                self.logger.info("Marker hasn't moved enough, skipping")
                return
            self.logger.info(f"Marker moved {dist_squared**0.5:.3f}m")

        # Update previous marker pose
        self._prev_marker_pose = Pose()
        self._prev_marker_pose.position.x = cal_marker_pose.position.x
        self._prev_marker_pose.position.y = cal_marker_pose.position.y
        self._prev_marker_pose.position.z = cal_marker_pose.position.z
        self._prev_marker_pose.orientation = cal_marker_pose.orientation

        # Transform pose
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_link, "camera_color_optical_frame", rclpy.time.Time()
            )

            # Transform the pose
            transformed_pose = do_transform_pose(cal_marker_pose, transform)

            # Publish the transformed pose
            stamped_pose = PoseStamped()
            stamped_pose.header.frame_id = self.base_link
            stamped_pose.header.stamp = self.get_clock().now().to_msg()
            stamped_pose.pose = transformed_pose
            # self.pose_pub.publish(stamped_pose)

            # Add offset for target pose
            target_pose = Pose()
            target_pose.position.x = transformed_pose.position.x
            target_pose.position.y = transformed_pose.position.y
            target_pose.position.z = transformed_pose.position.z + 0.01
            target_pose.orientation = transformed_pose.orientation

            # Publish target pose
            target_stamped = PoseStamped()
            target_stamped.header.frame_id = self.base_link
            target_stamped.header.stamp = self.get_clock().now().to_msg()
            target_stamped.pose = target_pose
            # self.target_pose_pub.publish(target_stamped)

            # Add to movement queue
            if not self.is_moving:
                self.pose_queue.put(target_pose)
                self.logger.info(
                    f"Added pose to queue. Queue size: {self.pose_queue.qsize()}"
                )

        except tf2_ros.LookupException as e:
            self.logger.error(f"Transform error: {e}")
        except Exception as e:
            self.logger.error(f"Error handling marker: {e}")


def main():
    rclpy.init()
    node = ArucoBoardFollower()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
