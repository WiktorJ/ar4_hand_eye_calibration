#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.node import ParameterType, ParameterDescriptor

from pymoveit2 import MoveIt2 # Replaces moveit_commander
# from moveit_msgs.msg import MoveItErrorCodes # Not directly used, pymoveit2 handles errors

from geometry_msgs.msg import PoseStamped
from easy_handeye2_msgs.srv import TakeSample, ComputeCalibration, SaveCalibration

import yaml
import numpy as np
import math
from transforms3d.quaternions import qinverse, qmult
from transforms3d.utils import vector_norm # Avoid conflict with np.linalg.norm

def calculate_pose_diff(pose1_stamped: PoseStamped, pose2_stamped: PoseStamped):
    """
    Calculates the translation and rotation difference between two PoseStamped messages.
    Assumes poses are in the same reference frame.
    """
    pos1 = np.array([pose1_stamped.pose.position.x, pose1_stamped.pose.position.y, pose1_stamped.pose.position.z])
    pos2 = np.array([pose2_stamped.pose.position.x, pose2_stamped.pose.position.y, pose2_stamped.pose.position.z])
    
    translation_diff = np.linalg.norm(pos1 - pos2)
    
    # transforms3d uses [w, x, y, z]
    q1 = np.array([pose1_stamped.pose.orientation.w, pose1_stamped.pose.orientation.x, 
                   pose1_stamped.pose.orientation.y, pose1_stamped.pose.orientation.z])
    q2 = np.array([pose2_stamped.pose.orientation.w, pose2_stamped.pose.orientation.x, 
                   pose2_stamped.pose.orientation.y, pose2_stamped.pose.orientation.z])
    
    # Normalize quaternions to be safe
    q1 /= vector_norm(q1)
    q2 /= vector_norm(q2)
    
    # Relative rotation: q_rel = q2 * q1_inverse (rotation from q1 to q2)
    q_rel = qmult(q2, qinverse(q1))
    
    # Angle from w component of relative rotation quaternion
    w_clipped = np.clip(q_rel[0], -1.0, 1.0) # q_rel[0] is w
    angle_rad = 2 * np.arccos(w_clipped)
    
    # Ensure shortest angle
    if angle_rad > np.pi:
        angle_rad = 2 * np.pi - angle_rad
        
    return translation_diff, angle_rad


class CalibrationOrchestrator(Node):
    def __init__(self):
        super().__init__('calibration_orchestrator')

        # Declare parameters
        self.declare_parameter('joint_states_yaml_path', '', ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description='Path to YAML file with joint states in degrees.'))
        self.declare_parameter('move_group_name', 'ar_manipulator', ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description='Name of the MoveIt move group.'))
        self.declare_parameter('min_translation_threshold_m', 0.01, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, description='Minimum translation (meters) to consider movement significant for taking a sample.'))
        self.declare_parameter('min_rotation_threshold_deg', 5.0, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, description='Minimum rotation (degrees) to consider movement significant.'))
        self.declare_parameter('handeye_calibration_name', 'ar4_calibration', ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description='Namespace for easy_handeye2 services.'))
        self.declare_parameter('num_joints', 6, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER, description='Expected number of joints for the robot.'))
        self.declare_parameter('planning_time_seconds', 10.0, ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, description='MoveIt planning time.'))
        self.declare_parameter('planning_attempts', 5, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER, description='MoveIt planning attempts.'))


        # Get parameters
        self.joint_states_yaml_path = self.get_parameter('joint_states_yaml_path').value
        self.move_group_name = self.get_parameter('move_group_name').value
        self.min_translation_threshold_m = self.get_parameter('min_translation_threshold_m').value
        min_rotation_threshold_deg = self.get_parameter('min_rotation_threshold_deg').value
        self.min_rotation_threshold_rad = math.radians(min_rotation_threshold_deg)
        self.handeye_calibration_name = self.get_parameter('handeye_calibration_name').value
        self.num_joints = self.get_parameter('num_joints').value
        planning_time_seconds = self.get_parameter('planning_time_seconds').value
        planning_attempts = self.get_parameter('planning_attempts').value

        if not self.joint_states_yaml_path:
            self.get_logger().error("'joint_states_yaml_path' parameter is not set or empty. Shutting down.")
            # Raise an exception or signal shutdown to the main function
            raise ValueError("'joint_states_yaml_path' parameter is not set.")

        # Initialize MoveIt2
        self.moveit2 = MoveIt2(
            node=self,
            group_name=self.move_group_name,
            # pymoveit2 will use default service/topic names for robot_description, etc.
            # It also infers joint_names, base_link_name, and end_effector_link if not provided.
        )
        # Wait for MoveIt2 to be ready (e.g. connected to services)
        # pymoveit2's constructor makes service calls, so it might need a brief spin or check.
        # However, typical usage doesn't show an explicit wait after MoveIt2 construction.
        # We assume it's ready or methods will block/fail appropriately.

        self.moveit2.set_planning_options(
            planning_time=planning_time_seconds,
            planning_attempts=planning_attempts
        )
        
        # Get frame names (pymoveit2 gets these on init, store them for logging consistency)
        # These calls might make service requests if not determined during MoveIt2 init.
        self.robot_base_frame = self.moveit2.base_link_name
        self.robot_effector_frame = self.moveit2.end_effector_link
        self.get_logger().info(f"MoveIt2 initialized for group '{self.move_group_name}' with base '{self.robot_base_frame}' and effector '{self.robot_effector_frame}'.")
        
        # Service Clients for easy_handeye2
        self.take_sample_client = self.create_client(TakeSample, f'/{self.handeye_calibration_name}/take_sample')
        self.compute_calibration_client = self.create_client(ComputeCalibration, f'/{self.handeye_calibration_name}/compute_calibration')
        self.save_calibration_client = self.create_client(SaveCalibration, f'/{self.handeye_calibration_name}/save_calibration')

        self.wait_for_services()

        # Load joint states
        self.target_joint_states_rad = self._load_joint_states_from_yaml()

        # State variables
        self.last_sampled_pose_stamped: PoseStamped = None
        self.samples_taken = 0
        
        self.get_logger().info("Calibration Orchestrator initialized.")

    def wait_for_services(self):
        services = {
            "take_sample": self.take_sample_client,
            "compute_calibration": self.compute_calibration_client,
            "save_calibration": self.save_calibration_client
        }
        for name, client in services.items():
            while not client.wait_for_service(timeout_sec=2.0):
                self.get_logger().info(f'Service {client.srv_name} not available, waiting again...')
            self.get_logger().info(f'Service {client.srv_name} is available.')


    def _load_joint_states_from_yaml(self):
        self.get_logger().info(f"Loading joint states from: {self.joint_states_yaml_path}")
        try:
            with open(self.joint_states_yaml_path, 'r') as file:
                data = yaml.safe_load(file)
                joint_states_degrees = data.get('joint_states_degrees', [])
                
                if not joint_states_degrees:
                    self.get_logger().error("No 'joint_states_degrees' found in YAML file or list is empty.")
                    raise ValueError("No 'joint_states_degrees' in YAML.")

                joint_states_rad = []
                for state_deg in joint_states_degrees:
                    if len(state_deg) != self.num_joints:
                        self.get_logger().error(f"Joint state {state_deg} has {len(state_deg)} values, expected {self.num_joints}.")
                        raise ValueError("Incorrect number of joints in a state.")
                    joint_states_rad.append([math.radians(j) for j in state_deg])
                
                self.get_logger().info(f"Loaded {len(joint_states_rad)} joint states.")
                return joint_states_rad
        except FileNotFoundError:
            self.get_logger().error(f"Joint states YAML file not found: {self.joint_states_yaml_path}")
            raise
        except Exception as e:
            self.get_logger().error(f"Error loading or parsing joint states YAML: {e}")
            raise

    def _get_current_effector_pose_stamped(self) -> PoseStamped:
        # Returns PoseStamped of the end-effector link in the planning frame (base_link_name)
        # pymoveit2.get_current_pose() returns pose in the base_link_name frame.
        return self.moveit2.get_current_pose()

    def _check_sufficient_movement(self, current_pose_stamped: PoseStamped) -> bool:
        if self.last_sampled_pose_stamped is None:
            self.get_logger().info("First sample, movement check not applicable.")
            return True # Always take the first sample

        # Ensure poses are in the same frame (should be, as both come from get_current_pose)
        if current_pose_stamped.header.frame_id != self.last_sampled_pose_stamped.header.frame_id:
            self.get_logger().warn(f"Frame ID mismatch for pose comparison: "
                                   f"{current_pose_stamped.header.frame_id} vs "
                                   f"{self.last_sampled_pose_stamped.header.frame_id}. Skipping movement check.")
            return True # Or handle transformation / error

        translation_diff, rotation_diff_rad = calculate_pose_diff(current_pose_stamped, self.last_sampled_pose_stamped)
        rotation_diff_deg = math.degrees(rotation_diff_rad)

        self.get_logger().info(f"Movement since last sample: Translation = {translation_diff:.4f} m, Rotation = {rotation_diff_deg:.2f} deg.")

        if translation_diff >= self.min_translation_threshold_m or rotation_diff_rad >= self.min_rotation_threshold_rad:
            self.get_logger().info("Sufficient movement detected.")
            return True
        else:
            self.get_logger().info("Insufficient movement since last sample. Skipping sample.")
            return False

    def _move_to_joint_state(self, joint_state_rad):
        self.get_logger().info(f"Moving to joint state: {[round(math.degrees(j),1) for j in joint_state_rad]} deg")
        
        # Plan
        self.get_logger().info("Planning kinematic path...")
        # joint_state_rad must be in the order of joint_names known to MoveIt2/SRDF.
        # pymoveit2.plan_kinematic_path uses options set by set_planning_options.
        trajectory = self.moveit2.plan_kinematic_path(joint_state_rad)

        if trajectory:
            # Planning time is not directly returned by plan_kinematic_path.
            self.get_logger().info(f"Plan successful. Executing trajectory...")
            # Execute
            # pymoveit2.execute with blocking=True waits for completion and returns status.
            execute_success = self.moveit2.execute(trajectory_msg=trajectory, blocking=True)
            
            if execute_success:
                self.get_logger().info("Movement execution successful.")
                # No explicit stop() or clear_pose_targets() needed as with moveit_commander.
                # execute(blocking=True) ensures completion. Targets are per-call.
                return True
            else:
                self.get_logger().error("Movement execution failed. Check MoveIt logs for details.")
                # pymoveit2's execute() logs error details if it fails.
                return False
        else:
            self.get_logger().error("Planning failed. Check MoveIt logs for details.")
            # pymoveit2.plan_kinematic_path returns None on failure.
            # Detailed error codes like from moveit_commander's plan() are not directly available here.
            return False

    def _call_service(self, client, request, service_name):
        if not client.service_is_ready():
            self.get_logger().error(f"Service {service_name} is not available.")
            return None
        
        self.get_logger().info(f"Calling {service_name} service...")
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0) # Spin self (the node)
        
        if future.done():
            try:
                response = future.result()
                self.get_logger().info(f"{service_name} call successful.")
                return response
            except Exception as e:
                self.get_logger().error(f"Service call {service_name} failed: {e}")
                return None
        else:
            self.get_logger().warn(f"{service_name} service call timed out.")
            return None

    def orchestrate_calibration_process(self):
        self.get_logger().info("Starting calibration orchestration process...")

        if not self.target_joint_states_rad:
            self.get_logger().error("No target joint states loaded. Aborting.")
            return

        for i, joint_state_rad in enumerate(self.target_joint_states_rad):
            self.get_logger().info(f"--- Processing calibration pose {i+1}/{len(self.target_joint_states_rad)} ---")
            
            if not self._move_to_joint_state(joint_state_rad):
                self.get_logger().warn(f"Failed to move to joint state {i+1}. Skipping this pose.")
                continue

            # Allow some time for TFs to stabilize if necessary, though MoveIt `execute` is blocking.
            # rclpy.spin_once(self, timeout_sec=0.5) # Or time.sleep(0.5)
            # self.get_clock().sleep_for(rclpy.duration.Duration(seconds=0.5)) # Pythonic sleep

            current_pose_stamped = self._get_current_effector_pose_stamped()
            if current_pose_stamped is None:
                self.get_logger().error("Failed to get current effector pose. Skipping sample.")
                continue
            
            self.get_logger().info(f"Current effector pose in {current_pose_stamped.header.frame_id}: "
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
                take_sample_resp = self._call_service(self.take_sample_client, take_sample_req, "TakeSample")

                if take_sample_resp and take_sample_resp.success:
                    self.samples_taken += 1
                    self.last_sampled_pose_stamped = current_pose_stamped
                    self.get_logger().info(f"Sample taken successfully. Total samples: {self.samples_taken}")
                    if self.samples_taken >= 3: # A common minimum for hand-eye calibration
                        compute_calib_req = ComputeCalibration.Request()
                        # Request can be empty
                        compute_calib_resp = self._call_service(self.compute_calibration_client, compute_calib_req, "ComputeCalibration")
                        if compute_calib_resp and compute_calib_resp.success:
                            self.get_logger().info("Calibration computed successfully.")
                            # Log calibration result if needed (from compute_calib_resp.calibration)
                            # For example:
                            # t = compute_calib_resp.calibration.transform.translation
                            # r = compute_calib_resp.calibration.transform.rotation
                            # self.get_logger().info(f"Computed Calibration Transform: T=({t.x:.4f}, {t.y:.4f}, {t.z:.4f}), Q=({r.x:.4f}, {r.y:.4f}, {r.z:.4f}, {r.w:.4f})")
                        else:
                            self.get_logger().warn("Failed to compute calibration after taking sample.")
                else:
                    self.get_logger().warn("Failed to take sample.")
            else:
                self.get_logger().info("Skipping sample due to insufficient movement.")
            
            self.get_logger().info(f"--- Finished processing pose {i+1} ---")


        if self.samples_taken > 0:
            self.get_logger().info("Calibration sequence finished. Attempting to save final calibration...")
            save_calib_req = SaveCalibration.Request() # Empty request
            save_calib_resp = self._call_service(self.save_calibration_client, save_calib_req, "SaveCalibration")
            if save_calib_resp and save_calib_resp.success:
                self.get_logger().info("Final calibration saved successfully.")
            else:
                self.get_logger().error("Failed to save final calibration.")
        else:
            self.get_logger().warn("No samples were taken. Calibration not saved.")
            
        self.get_logger().info("Calibration orchestration process complete.")


def main(args=None):
    rclpy.init(args=args)
    # moveit_commander.roscpp_initialize(args) # Not needed for pymoveit2

    orchestrator = None
    try:
        orchestrator = CalibrationOrchestrator()
        orchestrator.orchestrate_calibration_process() # This is a blocking call
    except ValueError as e: # Catch specific init errors
        if orchestrator:
            orchestrator.get_logger().error(f"Initialization failed: {e}")
        else:
            print(f"CalibrationOrchestrator initialization failed: {e}") # Logger not available yet
    except KeyboardInterrupt:
        if orchestrator: orchestrator.get_logger().info('Keyboard interrupt, shutting down.')
    except ExternalShutdownException:
        pass # Normal shutdown
    except Exception as e:
        if orchestrator: orchestrator.get_logger().error(f"An unexpected error occurred: {e}", exc_info=True)
        else: print(f"An unexpected error occurred in CalibrationOrchestrator: {e}")
    finally:
        if orchestrator:
            orchestrator.get_logger().info('Shutting down Calibration Orchestrator node.')
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
