#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Joint space control using joint trajectory action
"""
__author__ = "Rodrigo Muñoz"

import copy
from threading import Lock
import numpy as np
# ROS Core
import rospy
import actionlib
import math
# ROS Messages
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import (FollowJointTrajectoryAction, FollowJointTrajectoryGoal)

from scipy.interpolate import UnivariateSpline, splev, splrep

splines_dic = dict()
for letter in ["A","B","C","D"]:
    spline_file = np.load("spline_"+letter+".npz")
    splines_dic[letter] = spline_file["arr_0"]

def periodize(t,spline_parameters):
    real_t = t % 2
    print real_t
    return splev(real_t+4.081279275,spline_parameters) 

class LegSkill(object):
    """
    Base class for joint space control using joint trajectory action.
    """
    _type = "leg"

    # Class constants
    JOINT_NAMES_BASE = ["thigh_pitch_joint", "thigh_yaw_joint",
      "pelvis_joint", "tibia_joint", "ankle_joint",
      "phalange_joint"]
    """list of str: Joints names"""

    NUM_JOINTS = 6
    """int: Number of joints"""

    L_LEG = "l_leg"
    """str: Left arm name"""

    R_LEG = "r_leg"
    """str: Right arm name"""

    def __init__(self, arm_name):
        """
        Base class for joint space control using joint trajectory action

        Args:
            arm_name (str): Arm name (must be "L_LEG" or "R_LEG").

        Raises:
            TypeError: If `arm_name` is not a string.
            ValueError: If `arm_name` is not "L_LEG" or "R_LEG".
        """
        self._description = "Control using joint trajectory action"
        # Check arm name
        if not isinstance(arm_name, str):
            raise TypeError("arm_name must be a string")
        if not (arm_name == LegSkill.L_LEG or arm_name == LegSkill.R_LEG):
            raise ValueError("arm_name must be \"L_LEG\" or \"R_LEG\"")
        # Get arm name and side
        self.name = arm_name
        self.side = arm_name[0]
        # Get joint names
        self.joint_names = ["{0}_{1}".format(self.side, joint)
            for joint in LegSkill.JOINT_NAMES_BASE]
        # Arm topics
        self._joint_state_topic = "/joint_states"
        self._jta_topic = "/{0}_controller/follow_joint_trajectory".format(self.name)
        # Empty joint state message
        self._joint_state_lock = Lock()
        self._joint_state = JointState()
        self._joint_state.name = self.joint_names
        self._joint_state.position = [0.0]*LegSkill.NUM_JOINTS
        self._joint_state.velocity = [0.0]*LegSkill.NUM_JOINTS
        self._joint_state.effort = [0.0]*LegSkill.NUM_JOINTS
        # ROS clients (avoid linter warnings)
        self._joint_state_sub = None
        self._jta_client = None

    def _update_joint_state(self, msg):
        """
        Update joint positions.
        """
        i = 0
        with self._joint_state_lock:
            for j, joint in enumerate(self.joint_names):
                try:
                    i = msg.name.index(joint)
                except ValueError:
                    continue
                self._joint_state.position[j] = msg.position[i]
                self._joint_state.velocity[j] = msg.velocity[i]
                self._joint_state.effort[j] = msg.effort[i]
            self._joint_state.header = msg.header

    def get_joint_state(self):
        """
        Get current joint state.

        Returns:
            sensor_msgs.msg.JointState: Joint state.
        """
        # Acquire lock and return a complete copy
        with self._joint_state_lock:
            return copy.deepcopy(self._joint_state)

    def get_joint_names(self):
        """
        Get joint names.

        Returns:
            :obj:`list` of :obj:`str`: Joint names in order.
        """
        return copy.deepcopy(self.joint_names)

    def check(self, timeout=1.0):
        # Check client for joint trajectory action (JTA)
        jta_client = actionlib.SimpleActionClient(self._jta_topic, FollowJointTrajectoryAction)
        # Wait for the JTA server to start or exit
        if not jta_client.wait_for_server(timeout=rospy.Duration(timeout)):
            rospy.logerr("Joint trajectory action server for \"{0}\" not found".format(self.name))
            return False
        rospy.logdebug("Joint trajectory action server for \"{0}\" [OK]".format(self.name))
        # Check joint_states topic
        try:
            msg = rospy.wait_for_message(self._joint_state_topic, JointState, timeout=timeout)
        except rospy.ROSException:
            rospy.logerr("Topic \"{0}\" not already published".format(self._joint_state_topic))
            return False
        # Check arm joints in message
        for joint in self.joint_names:
            if not joint in msg.name:
                rospy.logerr("Topic \"{0}\" does not contain \"{1}\" joints".format(
                    self._joint_state_topic, self.name))
                return False
        rospy.logdebug("Topic \"{0}\" published [OK]".format(self._joint_state_topic))
     
        return True

    def setup(self):
        # Joint state subscriber
        self._joint_state_sub = rospy.Subscriber(self._joint_state_topic,
            JointState, self._update_joint_state)
        # Joint trajectory action (JTA)
        self._jta_client = actionlib.SimpleActionClient(self._jta_topic, FollowJointTrajectoryAction)
        # Get SRDF
        rospy.sleep(0.1)
        return True

    def shutdown(self):
        rospy.logwarn("Shutdown \"{0}\" skill, calling cancel goals...".format(self.name))
        # Cancel goals
        self._jta_client.cancel_all_goals()
        # Unregister subscriber
        self._joint_state_sub.unregister()
        return True

    def start(self):
        rospy.logdebug("Start \"{0}\" skill".format(self.name))
        return True

    def pause(self):
        rospy.logdebug("Pause \"{0}\" skill".format(self.name))
        self.stop()
        return True

    # Arm movement related methods
    def stop(self):
        """
        Cancel all goals.
        """
        rospy.logwarn("Stop \"{0}\", calling cancel goals...".format(self.name))
        self._jta_client.cancel_all_goals()

    def send_joint_goal(self, joint_goal, interval=3.0, segments=10):
        """
        Send joint goal reference to the arm.

        This function use linear interpolation between current position (obtained via joint_states topic)
        and joint goal.

        Args:
            joint_goal (list of float): Joint target configuration, must follow arm.get_joint_names() order.
            interval (float): Time interval between current position and joint goal.

        Examples:
            >>> arm.send_joint_goal([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # Send to home position
        """
        # Check joint state time stamp
        rospy.sleep(0.05)
        current_state = self.get_joint_state()
        if (rospy.Time.now() - current_state.header.stamp) > rospy.Duration(1.0):
            rospy.logerr("Current position has not been updated, check \"{}\" topic.".format(self._joint_state_topic))
            return
        # Create new goal
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.get_joint_names()
        dt = interval/segments
        t = 0.1
        inter_points = list()
        for i in range(LegSkill.NUM_JOINTS):
            # TODO(rorromr) Use parabolic interpolation
            inter_points.append(np.linspace(current_state.position[i], joint_goal[i], segments))
        print inter_points
        print "puntos!"
        for j in range(segments):
            point = JointTrajectoryPoint()
            point.positions = [joint[j] for joint in inter_points]
            t += dt
            point.time_from_start = rospy.Duration(t)
            goal.trajectory.points.append(point)
        # Send goal to JTA
        rospy.loginfo('Sending new goal for \"{0}\"'.format(self.name))
        self._jta_client.send_goal(goal)

    def wait_for_motion_done(self, timeout=0.0):
        """
        Blocks until gripper motion is done

        Args:
            timeout (float): Max time to block before returning. A zero timeout is interpreted as an infinite timeout.

        Returns:
            bool: True if the goal finished. False if the goal didn't finish within the allocated timeout.
        """
        rospy.loginfo('Waiting for \"{0}\" motion'.format(self.name))
        return self._jta_client.wait_for_result(rospy.Duration(timeout))

    def get_result(self):
        """
        Get movement result

        Returns:
            control_msgs.msg.FollowJointTrajectoryResult: If the goal finished.
            None: If the goal didn't finish.
        """
        return self._jta_client.get_result()

    ################################################

class LeftLegSkill(LegSkill):
    """Left arm control using using joint trajectory action"""
    _type = "l_leg"
    def __init__(self):
        super(LeftLegSkill, self).__init__(LegSkill.L_LEG)

class RightLegSkill(LegSkill):
    """Right arm control using using joint trajectory action"""
    _type = "r_leg"
    def __init__(self):
        super(RightLegSkill, self).__init__(LegSkill.R_LEG)
if __name__ == "__main__":
    rospy.init_node("grossi_saurio")
    thigh_angles = []
    tibia_angles = []
    ankle_angles = []
    phalange_angles = []
    time= np.arange(0,100,0.0001)
    time_c= np.arange(1,101,0.0001)
    thigh_angles = periodize(time,splines_dic["A"])
    tibia_angles = periodize(time,splines_dic["B"])
    ankle_angles = periodize(time,splines_dic["C"])
    phalange_angles = periodize(time_c,splines_dic["D"])
    thigh_angles_c = periodize(time_c,splines_dic["A"])
    tibia_angles_c = periodize(time_c,splines_dic["B"])
    ankle_angles_c = periodize(time_c,splines_dic["C"])
    phalange_angles_c = periodize(time_c,splines_dic["D"])

    for i,thetha in enumerate(thigh_angles):
        thigh_angles[i] = -1*(thetha+90)*(3.14/180) 
    for i,thetha in enumerate(tibia_angles):
        tibia_angles[i] = (180-thetha)*(3.14/180)     
    for i,thetha in enumerate(ankle_angles):
        ankle_angles[i] = (thetha-180)*(3.14/180) 
    for i,thetha in enumerate(phalange_angles):
        phalange_angles[i] = ((thetha-15)-180)*(3.14/180) 
    for i,thetha in enumerate(thigh_angles_c):
        thigh_angles_c[i] = -1*(thetha+90)*(3.14/180) 
    for i,thetha in enumerate(tibia_angles_c):
        tibia_angles_c[i] = (180-thetha)*(3.14/180)     
    for i,thetha in enumerate(ankle_angles_c):
        ankle_angles_c[i] = (thetha-180)*(3.14/180) 
    for i,thetha in enumerate(phalange_angles_c):
        phalange_angles_c[i] = ((thetha-15)-180)*(3.14/180) 
    l_leg = LeftLegSkill()
    r_leg = RightLegSkill()
    
    l_leg.setup()
    r_leg.setup()
    rate = rospy.Rate(100)
    while not rospy.is_shutdown():
        for i,thetha in enumerate(thigh_angles):
            l_leg.send_joint_goal([thigh_angles[i],0.0,0.0,tibia_angles[i],ankle_angles[i],phalange_angles[i]],segments=1)
            r_leg.send_joint_goal([thigh_angles_c[i],0.0,0.0,tibia_angles_c[i],ankle_angles_c[i],phalange_angles_c[i]],segments=1)
            l_leg.wait_for_motion_done()
            r_leg.wait_for_motion_done()
            print "ok"
            rate.sleep()


