#!/usr/bin/env python3
"""
A script to follow an aruco marker with a robot arm using PyMoveit2.
"""
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup, \
    MutuallyExclusiveCallbackGroup
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
        super().__init__("follow_aruco_marker")
        self.logger = self.get_logger()

        # The most important part: set QoS profile explicitly
        from rclpy.qos import QoSProfile, \
            ReliabilityPolicy, \
            HistoryPolicy, \
            DurabilityPolicy
        aruco_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=DurabilityPolicy.VOLATILE
        )

        # Create separate callback groups
        self.move_cb_group = ReentrantCallbackGroup()
        self.subscription_cb_group = ReentrantCallbackGroup()  # Changed to ReentrantCallbackGroup

        # ID of the aruco marker mounted on the robot
        self.marker_id = self.declare_parameter("marker_id",
                                                1).get_parameter_value().integer_value

        # Create TF buffer and listener first
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Create publishers
        self.pose_pub = self.create_publisher(PoseStamped, "/cal_marker_pose",
                                              1)
        self.target_pose_pub = self.create_publisher(PoseStamped,
                                                     "/follow_aruco_target_pose",
                                                     1)

        # Create direct subscription with explicit QoS
        self.logger.info(
            f"Creating subscription to /aruco_markers with QoS: {aruco_qos}")
        self.subscription = self.create_subscription(
            ArucoMarkers,
            "/aruco_markers",
            self.handle_aruco_markers,
            qos_profile=aruco_qos,
            callback_group=self.subscription_cb_group
        )

        # Add a simple timer to periodically check for markers
        self.create_timer(0.5, self.check_for_markers,
                          callback_group=self.subscription_cb_group)

        # Initialize MoveIt2 last
        self.arm_joint_names = [
            "camerajoint_1", "camerajoint_2", "camerajoint_3", "camerajoint_4",
            "camerajoint_5", "camerajoint_6"
        ]
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=self.arm_joint_names,
            base_link_name="camerabase_link",
            end_effector_name="cameralink_6",
            group_name="ar_manipulator",
            callback_group=self.move_cb_group,
        )
        self.moveit2.planner_id = "RRTConnectkConfigDefault"
        self.moveit2.max_velocity = 1.0
        self.moveit2.max_acceleration = 1.0

        self._prev_marker_pose = None
        self.is_moving = False
        self.last_aruco_msg_time = None

        # Add a timer to log subscription status
        self.create_timer(5.0, self.log_status,
                          callback_group=self.subscription_cb_group)

        self.logger.info("ArucoMarkerFollower node initialized")

    def check_for_markers(self):
        """Periodically log the time since last marker message"""
        if self.last_aruco_msg_time is not None:
            now = self.get_clock().now()
            elapsed = (now - self.last_aruco_msg_time).nanoseconds / 1e9
            if elapsed > 1.0:  # Only log if it's been more than a second
                self.logger.info(
                    f"It's been {elapsed:.2f} seconds since last ArUco marker message")

    def log_status(self):
        """Log current node status for debugging purposes"""
        self.logger.info(
            f"Node status - Is moving: {self.is_moving}, Has previous marker pose: {self._prev_marker_pose is not None}")
        self.logger.info(f"Waiting for ArUco marker with ID: {self.marker_id}")

        # Publish a debug message to check if publishers are working
        debug_pose = PoseStamped()
        debug_pose.header.frame_id = "camerabase_link"
        debug_pose.header.stamp = self.get_clock().now().to_msg()
        debug_pose.pose.position.x = 0.0
        debug_pose.pose.position.y = 0.0
        debug_pose.pose.position.z = 0.0
        debug_pose.pose.orientation.w = 1.0
        self.pose_pub.publish(debug_pose)

        # Check if aruco_node is publishing
        try:
            # Use ROS2 API to check topic statistics
            import subprocess
            result = subprocess.run(["ros2", "topic", "info", "/aruco_markers"],
                                    capture_output=True, text=True)
            self.logger.info(f"ArUco topic info: {result.stdout}")
        except Exception as e:
            self.logger.error(f"Failed to check topic info: {e}")

    def handle_aruco_markers(self, msg: ArucoMarkers):
        # Update the last message time
        self.last_aruco_msg_time = self.get_clock().now()

        self.logger.info(
            f"Received aruco_markers with {len(msg.marker_ids)} markers at timestamp {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")

        if self.is_moving:
            self.logger.info(
                "Still executing previous movement, skipping new marker update")
            return

        cal_marker_pose = None
        for i, marker_id in enumerate(msg.marker_ids):
            self.logger.info(
                f"Checking marker ID {marker_id} vs expected {self.marker_id}")
            if marker_id == self.marker_id:
                cal_marker_pose = msg.poses[i]
                self.logger.info(
                    f"Found marker {self.marker_id} at position: {cal_marker_pose.position.x}, {cal_marker_pose.position.y}, {cal_marker_pose.position.z}")
                break
            else:
                self.logger.info(
                    f"Detected unexpected marker with ID: {marker_id}")

        if cal_marker_pose is None:
            self.logger.info(
                f"No marker with ID {self.marker_id} found in message")
            return

        # Check if we need to move based on distance threshold
        if self._prev_marker_pose is not None:
            dist_squared = ((
                                        cal_marker_pose.position.x - self._prev_marker_pose.position.x) ** 2 +
                            (
                                        cal_marker_pose.position.y - self._prev_marker_pose.position.y) ** 2 +
                            (
                                        cal_marker_pose.position.z - self._prev_marker_pose.position.z) ** 2)

            # If the marker hasn't moved enough, skip following
            if dist_squared < 0.02 ** 2:
                self.logger.info(
                    f"Marker hasn't moved enough (dist={dist_squared ** 0.5:.4f}m), skipping movement")
                return
            else:
                self.logger.info(
                    f"Marker moved distance: {dist_squared ** 0.5:.4f}m, initiating follow")

        # Update previous marker position
        self._prev_marker_pose = Pose()
        self._prev_marker_pose.position.x = cal_marker_pose.position.x
        self._prev_marker_pose.position.y = cal_marker_pose.position.y
        self._prev_marker_pose.position.z = cal_marker_pose.position.z
        self._prev_marker_pose.orientation = cal_marker_pose.orientation

        # Transform pose to robot base frame
        try:
            transformed_pose = self._transform_pose(cal_marker_pose,
                                                    "camera_color_optical_frame",
                                                    "camerabase_link")
            self.logger.info(f"Following marker at pose: {transformed_pose}")

            # Set flag to prevent multiple movements at once
            self.is_moving = True
            self.move_to(transformed_pose)

        except tf2_ros.LookupException as e:
            self.logger.error(f"Error transforming pose: {e}")
            self.is_moving = False
            return
        except tf2_ros.TransformException as e:
            self.logger.error(f"Transform error: {e}")
            self.is_moving = False
            return
        except Exception as e:
            self.logger.error(f"Unexpected error in handling marker: {e}")
            self.is_moving = False
            return

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
        stamped_pose.header.stamp = self.get_clock().now().to_msg()
        stamped_pose.pose = transformed_pose
        self.pose_pub.publish(stamped_pose)

        # Create a copy of the pose to modify
        modified_pose = Pose()
        modified_pose.position.x = pose.position.x
        modified_pose.position.y = pose.position.y
        modified_pose.position.z = pose.position.z + 0.05
        modified_pose.orientation = pose.orientation

        transformed_pose = do_transform_pose(modified_pose, transform)

        stamped_pose = PoseStamped()
        stamped_pose.header.frame_id = target_frame
        stamped_pose.header.stamp = self.get_clock().now().to_msg()
        stamped_pose.pose = transformed_pose
        self.target_pose_pub.publish(stamped_pose)
        return transformed_pose

    def move_to(self, msg: Pose):
        try:
            pose_goal = PoseStamped()
            pose_goal.header.frame_id = "camerabase_link"
            pose_goal.header.stamp = self.get_clock().now().to_msg()
            pose_goal.pose = msg

            self.logger.info("Starting movement to target pose")
            self.moveit2.move_to_pose(pose=pose_goal)
            self.moveit2.wait_until_executed()
            self.logger.info("Finished move_to function")
        except Exception as e:
            self.logger.error(f"Error in move_to: {e}")
        finally:
            # Always clear the moving flag when done
            self.is_moving = False
            self.logger.info("Ready for next marker detection")


def main():
    rclpy.init()

    # Create the node
    node = ArucoMarkerFollower()

    # Create and configure the executor
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        node.logger.info("Starting executor...")
        executor.spin()
    except KeyboardInterrupt:
        node.logger.info("KeyboardInterrupt, shutting down...")
    except Exception as e:
        node.logger.error(f"Unhandled exception: {e}")
    finally:
        node.logger.info("Shutting down...")
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()