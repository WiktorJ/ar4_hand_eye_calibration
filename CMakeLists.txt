cmake_minimum_required(VERSION 3.5)
project(ar4_hand_eye_calibration)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(ros2_aruco_interfaces REQUIRED)
find_package(pymoveit2 REQUIRED)

install(DIRECTORY launch config rviz
  DESTINATION share/${PROJECT_NAME}
)

install (PROGRAMS 
  scripts/calibration_aruco_publisher.py
  scripts/follow_aruco_marker.py
  scripts/follow_aruco_board.py
  scripts/handeye_publisher.py
  scripts/visualize_aruco_marker.py
  scripts/calibration_orchestrator.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
