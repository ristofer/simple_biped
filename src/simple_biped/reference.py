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
from std_msgs.msg import Float64
from scipy.interpolate import UnivariateSpline, splev, splrep
import matplotlib.pyplot as plt
splines_dic = dict()
for letter in ["A","B","C","D"]:
    spline_file = np.load("spline_"+letter+".npz")
    splines_dic[letter] = spline_file["arr_0"]

def periodize(t,spline_parameters):
    real_t = t % 2
    print real_t
    return splev(real_t+4,spline_parameters) 

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
        self._joint_state_topic = "/joint_states_reference"
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
        self._joint_state_pub = None
        rospy.loginfo("Creating joint command publishers")
        #self._pub_joints={}
        #for j in self.joint_names:
         #   p=rospy.Publisher(j+"_position_controller/command",Float64,queue_size=10)
         #   self._pub_joints[j]=p


    def setup(self):

        self._joint_state_pub = rospy.Publisher(self._joint_state_topic,JointState,queue_size=10)

        rospy.sleep(0.1)
        return True
    def get_joint_names(self):
        """
        Get joint names.

        Returns:
            :obj:`list` of :obj:`str`: Joint names in order.
        """
        return copy.deepcopy(self.joint_names)
    # Arm movement related methods
    def set_joint_states(self,positions):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = self.get_joint_names()
        msg.position = positions
        self._joint_state_pub.publish(msg)
        print "set joint state"

    
    #def set_angles(self,angles):
     #   for i,angle in enumerate(angles):
     #       self._pub_joints[self.joint_names[i]].publish(angle)

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
    rospy.init_node("reference")
    thigh_angles = []
    tibia_angles = []
    ankle_angles = []
    phalange_angles = []
    time= np.arange(0,100,0.0005)
    time_c= np.arange(1,101,0.0005)
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
        phalange_angles[i] = ((thetha)-180)*(3.14/180) 
    for i,thetha in enumerate(thigh_angles_c):
        thigh_angles_c[i] = -1*(thetha+90)*(3.14/180) 
    for i,thetha in enumerate(tibia_angles_c):
        tibia_angles_c[i] = (180-thetha)*(3.14/180)     
    for i,thetha in enumerate(ankle_angles_c):
        ankle_angles_c[i] = (thetha-180)*(3.14/180) 
    for i,thetha in enumerate(phalange_angles_c):
        phalange_angles_c[i] = ((thetha)-180)*(3.14/180) 
    l_leg = LeftLegSkill()
    r_leg = RightLegSkill()

    l_leg.setup()
    r_leg.setup()
    #print r_leg.setup()
    print l_leg.get_joint_names()
    print r_leg.get_joint_names()
    print ankle_angles
    print len(thigh_angles)
    print len(phalange_angles)
    rate = rospy.Rate(10000)
    plt.plot(time, thigh_angles, 'g', lw=3)
    plt.show()
    while not rospy.is_shutdown():
        for i,thetha in enumerate(thigh_angles):
            l_leg.set_joint_states([thigh_angles[i],0.0,0.0,tibia_angles[i],ankle_angles[i],phalange_angles[i]])
            r_leg.set_joint_states([thigh_angles_c[i],0.0,0.0,tibia_angles_c[i],ankle_angles_c[i],phalange_angles_c[i]])
            #l_leg.set_angles([thigh_angles[i],0.0,0.0,tibia_angles[i],ankle_angles[i],phalange_angles[i]])
            #r_leg.set_angles([thigh_angles_c[i],0.0,0.0,tibia_angles_c[i],ankle_angles_c[i],phalange_angles_c[i]])
            print "ok"
            rospy.sleep(0.0005)


