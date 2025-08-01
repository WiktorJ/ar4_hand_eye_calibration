#! /usr/bin/env python3
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
from rclpy.node import ParameterType, ParameterDescriptor
from ros2_aruco_interfaces.msg import ArucoMarkers
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import PoseStamped


class CalibrationArucoPublisher(Node):
    """ROS2 node that listens to the aruco markers topic and publishes the 
    transform of the specific aruco marker for calibration to tf2.
    """

    def __init__(self):
        super().__init__("calibration_aruco_publisher")

        tracking_base_frame_p = self.declare_parameter(
            'tracking_base_frame',
            value="",
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )
        self.tracking_base_frame = tracking_base_frame_p.get_parameter_value().string_value
        tracking_marker_frame_p = self.declare_parameter(
            'tracking_marker_frame',
            value="",
            descriptor=ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )
        self.tracking_marker_frame = tracking_marker_frame_p.get_parameter_value().string_value
        self.declare_parameter(
            name="marker_type",
            value="board",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Type of marker to detect, 'board' or 'marker'.",
            ),
        )
        self.marker_type = (
            self.get_parameter("marker_type").get_parameter_value().string_value
        )
        self.marker_id = self.declare_parameter(
            "marker_id", 1).get_parameter_value().integer_value
        if self.marker_type == "marker":
            self.marker_subscription = self.create_subscription(ArucoMarkers,
                                                                "/aruco_markers",
                                                                self.handle_aruco_markers,
                                                                1)
        if self.marker_type == "board":
            self.board_subscription = self.create_subscription(PoseStamped,
                                                               "/aruco_board_pose",
                                                               self.handle_aruco_boards,
                                                               1)
        self.tf_broadcaster = TransformBroadcaster(self)

    def handle_aruco_markers(self, msg: ArucoMarkers):
        cal_marker_pose = None
        for i, marker_id in enumerate(msg.marker_ids):
            if marker_id == self.marker_id:
                cal_marker_pose = msg.poses[i]
                break

        if cal_marker_pose is None:
            return

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.tracking_base_frame
        t.child_frame_id = self.tracking_marker_frame

        t.transform.translation.x = cal_marker_pose.position.x
        t.transform.translation.y = cal_marker_pose.position.y
        t.transform.translation.z = cal_marker_pose.position.z
        t.transform.rotation = cal_marker_pose.orientation

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)

    def handle_aruco_boards(self, msg: PoseStamped):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.tracking_base_frame
        t.child_frame_id = self.tracking_marker_frame

        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation = msg.pose.orientation

        # Send the transformation
        self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = CalibrationArucoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == "__main__":
    main()
