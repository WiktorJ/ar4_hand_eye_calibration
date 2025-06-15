#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

_TF_PREFIX = 'camera'

class ArucoCharucoPoseEstimator(Node):
    """Node to estimate the pose of ArUco or ChArUco boards in the camera image.
    
    Run the following command to visualize the output:
    ros2 run image_view image_view image:=/aruco_image 

    Example launch command for a ChArUco board:
    ros2 run <your_package_name> visualize_aruco_charuco_boards.py --ros-args \
        -p board_type:="charuco" \
        -p charuco_squares_x:=5 \
        -p charuco_squares_y:=7 \
        -p charuco_square_length_m:=0.04 \
        -p charuco_marker_length_m:=0.025 \
        -p aruco_dictionary_name:="DICT_5X5_250"

    Example launch command for an ArUco board:
    ros2 run <your_package_name> visualize_aruco_charuco_boards.py --ros-args \
        -p board_type:="aruco" \
        -p aruco_markers_x:=5 \
        -p aruco_markers_y:=7 \
        -p aruco_marker_length_m:=0.04 \
        -p aruco_marker_separation_m:=0.01 \
        -p aruco_dictionary_name:="DICT_5X5_250"
    """

    def __init__(self):
        super().__init__('aruco_charuco_board_visualizer')

        self.bridge = CvBridge()

        # Declare parameters
        self.board_type = self.declare_parameter('board_type', 'charuco').get_parameter_value().string_value
        dictionary_name_param = self.declare_parameter('aruco_dictionary_name', 'DICT_5X5_250')
        dictionary_name = dictionary_name_param.get_parameter_value().string_value

        try:
            aruco_dict_id = getattr(cv2.aruco, dictionary_name)
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
        except AttributeError:
            self.get_logger().error(f"ArUco dictionary '{dictionary_name}' not found. Using DICT_5X5_250 as default.")
            # Fallback to a default dictionary if the specified one is not found
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)


        if self.board_type == 'aruco':
            markers_x = self.declare_parameter('aruco_markers_x', 5).get_parameter_value().integer_value
            markers_y = self.declare_parameter('aruco_markers_y', 7).get_parameter_value().integer_value
            marker_length_m = self.declare_parameter('aruco_marker_length_m', 0.04).get_parameter_value().double_value
            marker_separation_m = self.declare_parameter('aruco_marker_separation_m', 0.01).get_parameter_value().double_value
            self.board = cv2.aruco.GridBoard(
                size=[markers_x, markers_y],
                markerLength=marker_length_m,
                markerSeparation=marker_separation_m,
                dictionary=self.aruco_dict)
            self.axis_length = marker_length_m 
        elif self.board_type == 'charuco':
            squares_x = self.declare_parameter('charuco_squares_x', 5).get_parameter_value().integer_value
            squares_y = self.declare_parameter('charuco_squares_y', 7).get_parameter_value().integer_value
            square_length_m = self.declare_parameter('charuco_square_length_m', 0.04).get_parameter_value().double_value
            marker_length_m = self.declare_parameter('charuco_marker_length_m', 0.025).get_parameter_value().double_value
            self.board = cv2.aruco.CharucoBoard(
                size=[squares_x, squares_y],
                squareLength=square_length_m,
                markerLength=marker_length_m,
                dictionary=self.aruco_dict)
            self.axis_length = square_length_m / 2.0
        else:
            self.get_logger().error(f"Invalid board_type: {self.board_type}. Must be 'aruco' or 'charuco'. Shutting down.")
            # Consider raising an exception or handling this more gracefully
            raise ValueError("Invalid board_type specified.")
            
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(
            dictionary=self.aruco_dict, detectorParams=self.aruco_params)

        # Image subscriber
        self.image_subscription = self.create_subscription(
            Image,
            f'/camera/{_TF_PREFIX}/color/image_raw',
            self.image_callback,
            10)

        # CameraInfo subscriber for color camera
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            f'/camera/{_TF_PREFIX}/color/camera_info',
            self.camera_info_callback,
            10)
        self.camera_info_received = False

        self.camera_matrix = None
        self.dist_coeffs = None

        # Image publisher
        self.image_publisher = self.create_publisher(Image, '/aruco_image', 10)
        self.get_logger().info(f"ArucoCharucoPoseEstimator initialized for '{self.board_type}' board.")

    def camera_info_callback(self, msg):
        K = np.array(msg.k).reshape(3, 3)
        D = np.array(msg.d)
        self.camera_matrix = K
        self.dist_coeffs = D
        self.camera_info_received = True
        self.get_logger().info("Camera info received and parameters set.")
        self.destroy_subscription(self.camera_info_subscription)

    def image_callback(self, msg):
        if not self.camera_info_received:
            self.get_logger().warn("Waiting for camera info...")
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
            return

        corners, ids, rejectedImgPoints = self.aruco_detector.detectMarkers(cv_image)

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
            rvec, tvec = None, None # Initialize rvec and tvec

            if self.board_type == 'aruco':
                # For ArUco board, estimate pose using detected markers
                # The board object defines the world coordinates of the markers
                # objPoints is not used here as estimatePoseBoard calculates it internally
                pose_valid, rvec, tvec = cv2.aruco.estimatePoseBoard(
                    corners, ids, self.board, 
                    self.camera_matrix, self.dist_coeffs, 
                    rvec, tvec) # Pass rvec, tvec as None initially
                if pose_valid and rvec is not None and tvec is not None:
                    cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, 
                                      rvec, tvec, self.axis_length)

            elif self.board_type == 'charuco':
                # For ChArUco board, interpolate corners and then estimate pose
                if len(corners) > 0 and len(ids) > 0: # Ensure there are detected markers
                    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                        corners, ids, cv_image, self.board,
                        self.camera_matrix, self.dist_coeffs)
                    
                    if retval and charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 3:
                        # Draw interpolated Charuco corners
                        # cv2.aruco.drawDetectedCornersCharuco(cv_image, charuco_corners, charuco_ids) # Optional: visualize charuco corners

                        pose_valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                            charuco_corners, charuco_ids, self.board,
                            self.camera_matrix, self.dist_coeffs,
                            rvec, tvec) # Pass rvec, tvec as None initially
                        if pose_valid and rvec is not None and tvec is not None:
                            cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, 
                                              rvec, tvec, self.axis_length)
        try:
            overlay_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            self.image_publisher.publish(overlay_msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing image: {e}")


def main(args=None):
    rclpy.init(args=args)
    try:
        aruco_charuco_pose_estimator = ArucoCharucoPoseEstimator()
        rclpy.spin(aruco_charuco_pose_estimator)
    except ValueError as e: # Catch initialization errors like invalid board_type
        if rclpy.ok():
            node_for_logging = rclpy.create_node('aruco_charuco_error_logger') # Temporary node for logging if main node failed early
            node_for_logging.get_logger().error(f"Failed to initialize ArucoCharucoPoseEstimator: {e}")
            node_for_logging.destroy_node()
    except KeyboardInterrupt:
        pass
    finally:
        if 'aruco_charuco_pose_estimator' in locals() and rclpy.ok() and aruco_charuco_pose_estimator.executor is not None:
             aruco_charuco_pose_estimator.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
