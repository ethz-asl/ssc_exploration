#!/usr/bin/env python

import numpy as np
import rospy
import tf
import tf.transformations as tr
from sensor_msgs.msg import Image
from ssc_msgs.msg import SSCInput, SSCGrid


if __name__ == '__main__':
    rospy.init_node('ssc_input_adapter')
    listener = tf.TransformListener()
    pub = rospy.Publisher('ssc_input', SSCInput, queue_size=10)
    pub2 = rospy.Publisher('ssc_forward', SSCGrid, queue_size=10)

    world_frame = "world"

    def callback(img_msg): 
        # get depth camera pose wrt world.
        try:
            position, orientation = listener.lookupTransform(
                world_frame, img_msg.header.frame_id, img_msg.header.stamp)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        pose_matrix = tr.quaternion_matrix(orientation)
        pose_matrix[0:3, -1] = position
        
        # Create the msg.
        result = SSCInput()
        result.image = img_msg
        result.pose = np.array(pose_matrix).flatten().tolist()
        result.header = img_msg.header
        pub.publish(result)

    def callback2(ssc_msg): 
        # Republish the stack.
        pub2.publish(ssc_msg)
        
    sub= rospy.Subscriber('~depth_image', Image, callback)
    sub2= rospy.Subscriber('~ssc_output', SSCGrid, callback2)
    rospy.spin()
