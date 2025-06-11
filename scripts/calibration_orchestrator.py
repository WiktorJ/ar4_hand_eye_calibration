#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.node import ParameterType, ParameterDescriptor
from rclpy.time import Duration, Time
import tf2_ros
from tf2_ros import Buffer, TransformListener
from sensor_msgs.msg import JointState # Added import

from pymoveit2 import MoveIt2  # Replaces moveit_commander
# from moveit_msgs.msg import MoveItErrorCodes # Not directly used, pymoveit2 handles errors

from geometry_msgs.msg import PoseStamped
from easy_handeye2_msgs.srv import TakeSample, \
    ComputeCalibration, \
    SaveCalibration

import yaml
import numpy as np
import math
from transforms3d.quaternions import qinverse, qmult
from transforms3d.utils import vector_norm  # Avoid conflict with np.linalg.norm

# This is hard-coded because .rviz configs are not configurable and also have this prefix hard-coded
_TF_PREFIX = 'camera'


def calculate_pose_diff(pose1_stamped: PoseStamped, pose2_stamped: PoseStamped):
    """
    Calculates the translation and rotation difference between two PoseStamped messages.
    Assumes poses are in the same reference frame.
    """
    pos1 = np.array(
        [pose1_stamped.pose.position.x, pose1_stamped.pose.position.y,
         pose1_stamped.pose.position.z])
    pos2 = np.array(
        [pose2_stamped.pose.position.x, pose2_stamped.pose.position.y,
         pose2_stamped.pose.position.z])

    translation_diff = np.linalg.norm(pos1 - pos2)

    # transforms3d uses [w, x, y, z]
    q1 = np.array(
        [pose1_stamped.pose.orientation.w, pose1_stamped.pose.orientation.x,
         pose1_stamped.pose.orientation.y, pose1_stamped.pose.orientation.z])
    q2 = np.array(
        [pose2_stamped.pose.orientation.w, pose2_stamped.pose.orientation.x,
         pose2_stamped.pose.orientation.y, pose2_stamped.pose.orientation.z])

    # Normalize quaternions to be safe
    q1 /= vector_norm(q1)
    q2 /= vector_norm(q2)

    # Relative rotation: q_rel = q2 * q1_inverse (rotation from q1 to q2)
    q_rel = qmult(q2, qinverse(q1))

    # Angle from w component of relative rotation quaternion
    w_clipped = np.clip(q_rel[0], -1.0, 1.0)  # q_rel[0] is w
    angle_rad = 2 * np.arccos(w_clipped)

    # Ensure shortest angle
    if angle_rad > np.pi:
        angle_rad = 2 * np.pi - angle_rad

    return translation_diff, angle_rad


class CalibrationOrchestrator(Node):
    def __init__(self):
        super().__init__('calibration_orchestrator')

        arm_joint_names = [
            f"{_TF_PREFIX}joint_1", f"{_TF_PREFIX}joint_2",
            f"{_TF_PREFIX}joint_3", f"{_TF_PREFIX}joint_4",
            f"{_TF_PREFIX}joint_5", f"{_TF_PREFIX}joint_6"
        ]
        # Declare parameters
        self.declare_parameter('joint_states_yaml_path', '',
                               ParameterDescriptor(
                                   type=ParameterType.PARAMETER_STRING,
                                   description='Path to YAML file with joint states in degrees.'))
        self.declare_parameter('move_group_name', 'ar_manipulator',
                               ParameterDescriptor(
                                   type=ParameterType.PARAMETER_STRING,
                                   description='Name of the MoveIt move group.'))
        self.declare_parameter('robot_joint_names', arm_joint_names,
                               ParameterDescriptor(
                                   type=ParameterType.PARAMETER_STRING_ARRAY,
                                   description='List of joint names for the robot.'))
        self.declare_parameter('robot_base_link', f"{_TF_PREFIX}base_link",
                               ParameterDescriptor(
                                   type=ParameterType.PARAMETER_STRING,
                                   description='Name of the robot base link.'))
        self.declare_parameter('robot_end_effector_link', f"{_TF_PREFIX}link_6",
                               ParameterDescriptor(
                                   type=ParameterType.PARAMETER_STRING,
                                   description='Name of the robot end effector link.'))
        self.declare_parameter('tracking_base_frame', f"{_TF_PREFIX}_color_optical_frame", # Example default
                               ParameterDescriptor(
                                   type=ParameterType.PARAMETER_STRING,
                                   description='TF frame of the camera/tracking system base.'))
        self.declare_parameter('tracking_marker_frame', f"calibration_aruco",
                               ParameterDescriptor(
                                   type=ParameterType.PARAMETER_STRING,
                                   description='TF frame of the calibration marker.'))
        self.declare_parameter('min_translation_threshold_m', 0.01,
                               ParameterDescriptor(
                                   type=ParameterType.PARAMETER_DOUBLE,
                                   description='Minimum translation (meters) to consider movement significant for taking a sample.'))
        self.declare_parameter('min_rotation_threshold_deg', 5.0,
                               ParameterDescriptor(
                                   type=ParameterType.PARAMETER_DOUBLE,
                                   description='Minimum rotation (degrees) to consider movement significant.'))
        self.declare_parameter('handeye_calibration_name', 'easy_handeye2/calibration',
                               ParameterDescriptor(
                                   type=ParameterType.PARAMETER_STRING,
                                   description='Namespace for easy_handeye2 services.'))
        self.declare_parameter('num_joints', 6, ParameterDescriptor(
            type=ParameterType.PARAMETER_INTEGER,
            description='Expected number of joints for the robot.'))
        self.declare_parameter('planning_time_seconds', 10.0,
                               ParameterDescriptor(
                                   type=ParameterType.PARAMETER_DOUBLE,
                                   description='MoveIt planning time.'))
        self.declare_parameter('planning_attempts', 5, ParameterDescriptor(
            type=ParameterType.PARAMETER_INTEGER,
            description='MoveIt planning attempts.'))

        # Get parameters
        self.joint_states_yaml_path = self.get_parameter(
            'joint_states_yaml_path').value
        self.move_group_name = self.get_parameter('move_group_name').value
        self.robot_joint_names = self.get_parameter('robot_joint_names').value
        self.robot_base_link = self.get_parameter('robot_base_link').value
        self.robot_end_effector_link = self.get_parameter(
            'robot_end_effector_link').value
        self.tracking_base_frame = self.get_parameter('tracking_base_frame').value
        self.tracking_marker_frame = self.get_parameter('tracking_marker_frame').value
        self.min_translation_threshold_m = self.get_parameter(
            'min_translation_threshold_m').value
        min_rotation_threshold_deg = self.get_parameter(
            'min_rotation_threshold_deg').value
        self.min_rotation_threshold_rad = math.radians(
            min_rotation_threshold_deg)
        self.handeye_calibration_name = self.get_parameter(
            'handeye_calibration_name').value
        self.num_joints = self.get_parameter('num_joints').value
        planning_time_seconds = self.get_parameter(
            'planning_time_seconds').value
        planning_attempts = self.get_parameter('planning_attempts').value

        if not self.joint_states_yaml_path:
            self.get_logger().error(
                "'joint_states_yaml_path' parameter is not set or empty. Shutting down.")
            # Raise an exception or signal shutdown to the main function
            raise ValueError("'joint_states_yaml_path' parameter is not set.")

        # Initialize MoveIt2
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=self.robot_joint_names,
            base_link_name=self.robot_base_link,
            end_effector_name=self.robot_end_effector_link,
            group_name=self.move_group_name
        )
        # Wait for MoveIt2 to be ready (e.g. connected to services)
        # pymoveit2's constructor makes service calls, so it might need a brief spin or check.
        # However, typical usage doesn't show an explicit wait after MoveIt2 construction.
        # We assume it's ready or methods will block/fail appropriately.

        # Set planning options using properties
        self.moveit2.allowed_planning_time = planning_time_seconds
        self.moveit2.num_planning_attempts = planning_attempts

        # Get frame names (these should match the parameters passed to MoveIt2)
        self.robot_base_frame = self.moveit2.base_link_name
        self.robot_effector_frame = self.moveit2.end_effector_name
        self.get_logger().info(
            f"MoveIt2 initialized for group '{self.move_group_name}' with base '{self.robot_base_frame}' and effector '{self.robot_effector_frame}'.")
        self.get_logger().info(
            f"MoveIt2 using joint names: {self.moveit2.joint_names}")

        # TF Buffer and Listener
        self.tf_buffer = Buffer(node=self)
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # Wait for the robot's own kinematic chain TF to be ready.
        # This is essential for MoveIt2 to function correctly (e.g., get_current_pose).
        if not self._wait_for_robot_kinematic_chain_tf():
            self.get_logger().error(
                "Shutdown requested or failed to get robot kinematic chain TF. Aborting initialization.")
            # Propagate shutdown or raise a specific error to halt execution
            raise rclpy.executors.ExternalShutdownException()


        # Service Clients for easy_handeye2
        # We create them here, but wait for them later, after the first move.
        self.take_sample_client = self.create_client(TakeSample,
                                                     f'/{self.handeye_calibration_name}/take_sample')
        self.compute_calibration_client = self.create_client(ComputeCalibration,
                                                             f'/{self.handeye_calibration_name}/compute_calibration')
        self.save_calibration_client = self.create_client(SaveCalibration,
                                                          f'/{self.handeye_calibration_name}/save_calibration')

        # NOTE: self.wait_for_services() is MOVED to _wait_for_tracking_tf_and_services()
        # and called after the first robot move in orchestrate_calibration_process().

        # Load joint states
        self.target_joint_states_rad = self._load_joint_states_from_yaml()

        # State variables
        self.last_sampled_pose_stamped: PoseStamped = None
        self.samples_taken = 0

        self.get_logger().info("Calibration Orchestrator initialized.")

    def _wait_for_robot_kinematic_chain_tf(self) -> bool:
        self.get_logger().info("Waiting for robot kinematic chain TF (base to effector) to be available...")
        robot_frames_ok = False
        while not robot_frames_ok and rclpy.ok():
            try:
                self.tf_buffer.lookup_transform(
                    self.robot_base_link, self.robot_end_effector_link,
                    Time(), timeout=Duration(seconds=1.0))
                robot_frames_ok = True
            except tf2_ros.TransformException as ex:
                self.get_logger().debug( # Changed to debug to reduce noise if it takes a few tries
                    f"Waiting for robot transform {self.robot_base_link} -> {self.robot_end_effector_link}: {ex}")
                self.get_clock().sleep_for(Duration(seconds=1.0))

        if not rclpy.ok():
            self.get_logger().warn("Shutdown requested while waiting for robot kinematic chain TF.")
            return False
        
        self.get_logger().info(
            f"Robot transform {self.robot_base_link} -> {self.robot_end_effector_link} is available.")
        return True

    def _wait_for_tracking_tf_and_services(self) -> bool:
        self.get_logger().info("Waiting for tracking system TF (tracking base to marker) to be available...")
        tracking_frames_ok = False
        while not tracking_frames_ok and rclpy.ok():
            try:
                self.tf_buffer.lookup_transform(
                    self.tracking_base_frame, self.tracking_marker_frame,
                    Time(), timeout=Duration(seconds=10.0))
                tracking_frames_ok = True
                self.get_logger().info( # Log success immediately
                    f"Tracking transform {self.tracking_base_frame} -> {self.tracking_marker_frame} is available.")
            except tf2_ros.TransformException as ex:
                self.get_logger().info(
                    f"Waiting for tracking transform {self.tracking_base_frame} -> {self.tracking_marker_frame}: {ex}")
                # try:
                #     all_frames = self.tf_buffer.all_frames_as_string()
                #     self.get_logger().info(f"Orchestrator TF Buffer Contents (while waiting for tracking TF): {all_frames}")
                # except Exception as e_diag:
                #     self.get_logger().error(f"Failed to get all_frames_as_string for diagnostics: {e_diag}")
                
                if not rclpy.ok(): # Check if shutdown was requested before attempting to spin
                    self.get_logger().warn("Shutdown detected before spin_once in TF wait loop.")
                    break # Exit the while loop
                try:
                    # Spin to allow TF updates and other callbacks to be processed.
                    # The TransformListener has its own thread, but this ensures the node's main queue is active.
                    rclpy.spin_once(self, timeout_sec=2.0)
                except ExternalShutdownException:
                    self.get_logger().warn("Shutdown requested during spin_once in _wait_for_tracking_tf.")
                    break # Exit the while loop if shutdown occurs during spin

        if not tracking_frames_ok:
            if not rclpy.ok(): # If loop exited due to shutdown
                self.get_logger().warn(
                    f"Shutdown requested while waiting for tracking TF ({self.tracking_base_frame} -> {self.tracking_marker_frame}), TF not yet found.")
            else: # If loop exited for other reasons (e.g. max attempts if implemented) but rclpy still ok
                self.get_logger().error(
                    f"Failed to find tracking transform {self.tracking_base_frame} -> {self.tracking_marker_frame} after loop completion.")
            return False
        
        # If we reach here, tracking_frames_ok is True.
        # rclpy.ok() should also be true, otherwise the 'if not tracking_frames_ok:' block above would have caught it
        # if the loop exited due to !rclpy.ok() before TF was found.
        
        self.get_logger().info("Tracking TF confirmed. Now waiting for easy_handeye2 services...")
        self.wait_for_services() 

        if not rclpy.ok(): # Check if shutdown occurred during wait_for_services
            self.get_logger().warn("Shutdown requested while waiting for easy_handeye2 services.")
            return False
            
        self.get_logger().info("Tracking TF and easy_handeye2 services are ready.")
        return True

    def wait_for_services(self):
        services = {
            "take_sample": self.take_sample_client,
            "compute_calibration": self.compute_calibration_client,
            "save_calibration": self.save_calibration_client
        }
        for name, client in services.items():
            while not client.wait_for_service(timeout_sec=2.0):
                self.get_logger().info(
                    f'Service {client.srv_name} not available, waiting again...')
            self.get_logger().info(f'Service {client.srv_name} is available.')

    def _load_joint_states_from_yaml(self):
        self.get_logger().info(
            f"Loading joint states from: {self.joint_states_yaml_path}")
        try:
            with open(self.joint_states_yaml_path, 'r') as file:
                data = yaml.safe_load(file)
                joint_states_degrees = data.get('joint_states_degrees', [])

                if not joint_states_degrees:
                    self.get_logger().error(
                        "No 'joint_states_degrees' found in YAML file or list is empty.")
                    raise ValueError("No 'joint_states_degrees' in YAML.")

                joint_states_rad = []
                for state_deg in joint_states_degrees:
                    if len(state_deg) != self.num_joints:
                        self.get_logger().error(
                            f"Joint state {state_deg} has {len(state_deg)} values, expected {self.num_joints}.")
                        raise ValueError(
                            "Incorrect number of joints in a state.")
                    joint_states_rad.append(
                        [math.radians(j) for j in state_deg])

                self.get_logger().info(
                    f"Loaded {len(joint_states_rad)} joint states.")
                return joint_states_rad
        except FileNotFoundError:
            self.get_logger().error(
                f"Joint states YAML file not found: {self.joint_states_yaml_path}")
            raise
        except Exception as e:
            self.get_logger().error(
                f"Error loading or parsing joint states YAML: {e}")
            raise

    def _get_current_effector_pose_stamped(self) -> PoseStamped:
        # Returns PoseStamped of the end-effector link in the planning frame (base_link_name)
        
        # Get the full current joint state from MoveIt2, which might include non-group joints
        full_joint_state: JointState = self.moveit2.joint_state
        if full_joint_state is None:
            self.get_logger().error("Could not get current joint_state from MoveIt2.")
            return None

        # Filter to include only the joints relevant to the MoveIt2 instance's planning group
        filtered_joint_state = JointState()
        filtered_joint_state.header = full_joint_state.header # Retain header for timestamp consistency
        
        relevant_joint_names = self.moveit2.joint_names
        relevant_joint_positions = []
        
        # Create a mapping from name to position for efficient lookup
        full_joint_positions_map = dict(zip(full_joint_state.name, full_joint_state.position))
        
        for name in relevant_joint_names:
            if name in full_joint_positions_map:
                filtered_joint_state.name.append(name)
                relevant_joint_positions.append(full_joint_positions_map[name])
            else:
                self.get_logger().warn(f"Joint '{name}' from MoveIt2 joint_names not found in current full_joint_state. Skipping.")
        
        if len(filtered_joint_state.name) != len(relevant_joint_names):
            self.get_logger().error("Mismatch in relevant joints found in full_joint_state. Cannot reliably compute FK.")
            return None
            
        filtered_joint_state.position = relevant_joint_positions
        # Velocity and effort are not strictly needed for FK but can be populated if available and desired
        # For FK, only names and positions are essential.

        # self.moveit2.compute_fk() with no arguments defaults to current joint state and end-effector.
        # We now pass the filtered joint state.
        return self.moveit2.compute_fk(joint_state=filtered_joint_state)

    def _check_sufficient_movement(self,
                                   current_pose_stamped: PoseStamped) -> bool:
        if self.last_sampled_pose_stamped is None:
            self.get_logger().info(
                "First sample, movement check not applicable.")
            return True  # Always take the first sample

        # Ensure poses are in the same frame (should be, as both come from get_current_pose)
        if current_pose_stamped.header.frame_id != self.last_sampled_pose_stamped.header.frame_id:
            self.get_logger().warn(f"Frame ID mismatch for pose comparison: "
                                   f"{current_pose_stamped.header.frame_id} vs "
                                   f"{self.last_sampled_pose_stamped.header.frame_id}. Skipping movement check.")
            return True  # Or handle transformation / error

        translation_diff, rotation_diff_rad = calculate_pose_diff(
            current_pose_stamped, self.last_sampled_pose_stamped)
        rotation_diff_deg = math.degrees(rotation_diff_rad)

        self.get_logger().info(
            f"Movement since last sample: Translation = {translation_diff:.4f} m, Rotation = {rotation_diff_deg:.2f} deg.")

        if translation_diff >= self.min_translation_threshold_m or rotation_diff_rad >= self.min_rotation_threshold_rad:
            self.get_logger().info("Sufficient movement detected.")
            return True
        else:
            self.get_logger().info(
                "Insufficient movement since last sample. Skipping sample.")
            return False

    def _move_to_joint_state(self, joint_state_rad):
        self.get_logger().info(
            f"Moving to joint state: {[round(math.degrees(j), 1) for j in joint_state_rad]} deg")

        # Plan
        self.get_logger().info("Moving to configuration...")
        # joint_state_rad must be in the order of joint_names known to MoveIt2.
        # The joint_names are already set during MoveIt2 initialization.
        # move_to_configuration handles planning and execution.
        # It uses the planning options (time, attempts) set on the moveit2 object.
        self.moveit2.move_to_configuration(
            joint_positions=joint_state_rad,
            # joint_names can be omitted if using the default joint_names from init
        )

        # Wait for the execution to complete and get the result
        # wait_until_executed returns True on success, False on failure.
        execution_successful = self.moveit2.wait_until_executed()

        if execution_successful:
            self.get_logger().info("Movement execution successful.")
            return True
        else:
            self.get_logger().error(
                "Movement execution failed. Check MoveIt logs for details. Last error code: {}".format(
                    self.moveit2.get_last_execution_error_code()
                )
            )
            return False

    def _call_service(self, client, request, service_name):
        if not client.service_is_ready():
            self.get_logger().error(f"Service {service_name} is not available.")
            return None

        self.get_logger().info(f"Calling {service_name} service...")
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future,
                                         timeout_sec=10.0)  # Spin self (the node)

        if future.done():
            try:
                response = future.result()
                self.get_logger().info(f"{service_name} call successful.")
                return response
            except Exception as e:
                self.get_logger().error(
                    f"Service call {service_name} failed: {e}")
                return None
        else:
            self.get_logger().warn(f"{service_name} service call timed out.")
            return None

    def orchestrate_calibration_process(self):
        self.get_logger().info("Starting calibration orchestration process...")

        if not self.target_joint_states_rad:
            self.get_logger().error("No target joint states loaded. Aborting.")
            return

        # --- Initial Setup: Move to first pose, then wait for TFs/services ---
        self.get_logger().info("Performing initial setup: moving to the first calibration pose...")
        first_joint_state_rad = self.target_joint_states_rad[0]
        if not self._move_to_joint_state(first_joint_state_rad):
            self.get_logger().error(
                "Failed to move to the initial joint state. Aborting calibration process.")
            return

        self.get_logger().info(
            "Initial move complete. Now ensuring tracking TF and services are ready.")
        if not self._wait_for_tracking_tf_and_services():
            self.get_logger().error(
                "Tracking TF or services did not become ready after initial move. Aborting calibration process.")
            return
        # --- End of Initial Setup ---

        self.get_logger().info(
            "Initial setup complete. Starting main calibration sequence.")
        for i, joint_state_rad in enumerate(self.target_joint_states_rad):
            self.get_logger().info(
                f"--- Processing calibration pose {i + 1}/{len(self.target_joint_states_rad)} ---")

            if not self._move_to_joint_state(joint_state_rad):
                self.get_logger().warn(
                    f"Failed to move to joint state {i + 1}. Skipping this pose.")
                continue

            # Allow some time for TFs to stabilize if necessary, though MoveIt `execute` is blocking.
            rclpy.spin_once(self, timeout_sec=3.0)
            # self.get_clock().sleep_for(rclpy.duration.Duration(seconds=0.5)) # Pythonic sleep

            current_pose_stamped = self._get_current_effector_pose_stamped()
            if current_pose_stamped is None:
                self.get_logger().error(
                    "Failed to get current effector pose. Skipping sample.")
                continue

            self.get_logger().info(
                f"Current effector pose in {current_pose_stamped.header.frame_id}: "
                f"Pos(x,y,z): ({current_pose_stamped.pose.position.x:.3f}, "
                f"{current_pose_stamped.pose.position.y:.3f}, "
                f"{current_pose_stamped.pose.position.z:.3f}), "
                f"Ori(x,y,z,w): ({current_pose_stamped.pose.orientation.x:.3f}, "
                f"{current_pose_stamped.pose.orientation.y:.3f}, "
                f"{current_pose_stamped.pose.orientation.z:.3f}, "
                f"{current_pose_stamped.pose.orientation.w:.3f})")

            if self._check_sufficient_movement(current_pose_stamped):
                take_sample_req = TakeSample.Request()
                # Request can be empty, easy_handeye2 takes sample based on current TF state
                take_sample_resp = self._call_service(self.take_sample_client,
                                                      take_sample_req,
                                                      "TakeSample")

                if take_sample_resp:  # If TakeSample service call returned a response, assume success
                    self.samples_taken += 1
                    self.last_sampled_pose_stamped = current_pose_stamped
                    self.get_logger().info(
                        f"Sample taken successfully. Total samples: {self.samples_taken}")
                    if self.samples_taken >= 3:  # A common minimum for hand-eye calibration
                        compute_calib_req = ComputeCalibration.Request()
                        # Request can be empty
                        compute_calib_resp = self._call_service(
                            self.compute_calibration_client, compute_calib_req,
                            "ComputeCalibration")
                        if compute_calib_resp: # If ComputeCalibration service call returned a response, assume success
                            self.get_logger().info(
                                "Calibration computed successfully.")
                            t = compute_calib_resp.calibration.transform.translation
                            r = compute_calib_resp.calibration.transform.rotation
                            self.get_logger().info(f"Computed Calibration Transform: T=({t.x:.4f}, {t.y:.4f}, {t.z:.4f}), Q=({r.x:.4f}, {r.y:.4f}, {r.z:.4f}, {r.w:.4f})")
                        else:
                            self.get_logger().warn(
                                "Failed to compute calibration after taking sample.")
                else:
                    self.get_logger().warn("Failed to take sample.")
            else:
                self.get_logger().info(
                    "Skipping sample due to insufficient movement.")

            self.get_logger().info(f"--- Finished processing pose {i + 1} ---")

        if self.samples_taken > 0:
            self.get_logger().info(
                "Calibration sequence finished. Attempting to save final calibration...")
            save_calib_req = SaveCalibration.Request()  # Empty request
            save_calib_resp = self._call_service(self.save_calibration_client,
                                                 save_calib_req,
                                                 "SaveCalibration")
            if save_calib_resp: # If SaveCalibration service call returned a response, assume success
                self.get_logger().info("Final calibration saved successfully.")
            else:
                self.get_logger().error("Failed to save final calibration.")
        else:
            self.get_logger().warn(
                "No samples were taken. Calibration not saved.")

        self.get_logger().info("Calibration orchestration process complete.")


def main(args=None):
    rclpy.init(args=args)
    # moveit_commander.roscpp_initialize(args) # Not needed for pymoveit2

    orchestrator = None
    try:
        orchestrator = CalibrationOrchestrator()
        orchestrator.orchestrate_calibration_process()  # This is a blocking call
    except ValueError as e:  # Catch specific init errors
        if orchestrator:
            orchestrator.get_logger().error(f"Initialization failed: {e}")
        else:
            print(
                f"CalibrationOrchestrator initialization failed: {e}")  # Logger not available yet
    except KeyboardInterrupt:
        if orchestrator: orchestrator.get_logger().info(
            'Keyboard interrupt, shutting down.')
    except ExternalShutdownException:
        pass  # Normal shutdown
    except Exception as e:
        if orchestrator:
            orchestrator.get_logger().error(
                f"An unexpected error occurred: {e}")
        else:
            print(
                f"An unexpected error occurred in CalibrationOrchestrator: {e}")
    finally:
        if orchestrator:
            orchestrator.get_logger().info(
                'Shutting down Calibration Orchestrator node.')
            # Consider stopping any ongoing MoveIt2 actions if orchestrator is destroyed mid-operation
            # For example, if orchestrate_calibration_process could be interrupted:
            # if hasattr(orchestrator, 'moveit2'):
            #    orchestrator.moveit2.cancel_all_goals() # Optional: ensure robot stops if shutdown is abrupt
            orchestrator.destroy_node()
        # moveit_commander.roscpp_shutdown() # Not needed for pymoveit2
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
