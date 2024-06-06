#!/usr/bin/env python3
import os

# Network dependencies
import torch
import argparse
import numpy as np
from torch.autograd import Variable

# ROS dependencies
import rospy
from sensor_msgs.msg import Image
import tf.transformations as tr
import tf
from cv_bridge import CvBridge

# local imports
from models import make_model
from utils import utils
from ssc_msgs.msg import SSCGrid


class ROSInfer:
    def __init__(self):
        self._load_arguments()
        self.net = make_model(self.args.model, num_classes=12)
        self.input_topic_name = self.args.input_topic_name
        self.output_topic_name = self.args.output_topic_name
        self.world_frame = self.args.world_frame
        self.listener = tf.TransformListener()
        self.ssc_pub = rospy.Publisher(self.output_topic_name, SSCGrid, queue_size=10)
        self.bridge = CvBridge()

    def start(self):
        """
        Loads SSC Network model and start listening to depth images.
        """
        # load pretrained model
        self.load_network()
        self.depth_img_subscriber = rospy.Subscriber(
            self.input_topic_name, Image, self.callback, queue_size=1)
        print("SSC Inference setup successfully!")

    def callback(self, depth_image):
        """
        Receive a Depth image from the simulation, voxelize the depthmap as TSDF, 2D to 3D mapping
        and perform inference using 3D CNN. Publish the results as SSCGrid Message.
        """

        # get depth camera pose wrt odom
        try:
            position, orientation = self.listener.lookupTransform(
                self.world_frame, depth_image.header.frame_id, depth_image.header.stamp)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # parse depth image
        cv_image = self.bridge.imgmsg_to_cv2(
            depth_image, desired_encoding='passthrough')

        # prepare pose matrix
        pose_matrix = tr.quaternion_matrix(orientation)
        pose_matrix[0:3, -1] = position

        vox_origin, rgb, depth, tsdf, position, occupancy_grid = self._load_data_from_depth_image(
            cv_image, pose_matrix)
        x_depth = Variable(depth.float()).to(self.device)
        position = position.long().to(self.device)

        # x_tsdf = Variable(tsdf.float()).to(self.device)
        # y_pred = self.net(x_depth=x_depth, x_tsdf=x_tsdf, p=position)
        y_pred = self.net(x_depth=x_depth, p=position)


        scores = torch.nn.Softmax(dim=0)(y_pred.squeeze())
        # Threshold max probs for encoding free space
        max_prob = 1.0 - 1e-8
        scores[scores> max_prob] = max_prob
        free_space_confidence = scores[0]
        
        # Encode free space scores along with class id.
        preds = torch.argmax(scores, dim=0).cpu().numpy() + free_space_confidence.detach().cpu().numpy()

        # setup message
        msg = SSCGrid()
        msg.header = depth_image.header
        msg.data = preds.reshape(-1).astype(np.float32).tolist()

        msg.origin_x = vox_origin[0]
        msg.origin_y = vox_origin[1]
        msg.origin_z = vox_origin[2]
        msg.frame = 'odom'

        msg.width = preds.shape[0]
        msg.height = preds.shape[1]
        msg.depth = preds.shape[2]

        # publish message
        self.ssc_pub.publish(msg)

    def _load_data_from_depth_image(self, depth, cam_pose, max_depth=8, cam_k=[[320, 0, 320], [0, 320, 240], [0, 0, 1]]):
        """
        Takes a depth map, pose as input and outputs the 3D voxeloccupancy, 2D to 3D mapping and TSDF grid.
        """
        rgb = None
        depth_npy = np.array(depth)

        # discard inf points
        depth_npy[depth_npy > max_depth] = depth_npy.min()

        # get voxel grid origin
        vox_origin = utils.get_origin_from_depth_image(
            depth_npy, cam_k, cam_pose)

        # compute tsdf for the voxel grid from depth camera
        vox_tsdf, depth_mapping_idxs, voxel_occupancy = utils.compute_tsdf(
            depth_npy, vox_origin, cam_k, cam_pose)

        return vox_origin, rgb, torch.as_tensor(depth_npy).unsqueeze(0).unsqueeze(0), torch.as_tensor(vox_tsdf).unsqueeze(0), torch.as_tensor(depth_mapping_idxs).unsqueeze(0).unsqueeze(0), torch.as_tensor(voxel_occupancy.transpose(2, 1, 0)).unsqueeze(0)

    def load_network(self):
        """
        Loads a pretrained model for inference
        """
        if torch.cuda.is_available():
            print("CUDA device found!".format(torch.cuda.device_count()))
            self.device = torch.device('cuda')
        else:
            print("Using CPU!")
            self.device = torch.device('cpu')

        if os.path.isfile(self.args.resume):
            print("=> loading checkpoint '{}'".format(self.args.resume))
            cp_states = torch.load(
                self.args.resume, map_location=torch.device('cpu'))
            self.net.load_state_dict(cp_states['state_dict'], strict=True)
        else:
            raise Exception(
                "=> NO checkpoint found at '{}'".format(self.args.resume))
        self.net = self.net.to(self.device)
        self.net.eval()

    def _load_arguments(self):
        parser = argparse.ArgumentParser(description='PyTorch SSC Inference')
        parser.add_argument('--input_topic_name', type=str, default='/airsim_drone/Depth_cam',
                            help='depth image topic to subscribe to (default: /airsim_drone/Depth_cam)')
        parser.add_argument('--output_topic_name', type=str, default='/ssc',
                            help='Output topic name to publish to (default: /ssc)')
        parser.add_argument('--world_frame', type=str, default='/odom',
                            help='world frame name (default: /odom)')
        parser.add_argument('--model', type=str, default='palnet', choices=['ddrnet', 'palnet', 'palnet_ours'],
                            help='model name (default: palnet)')

        parser.add_argument('--resume', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        args = parser.parse_args()

        # use argparse arguments as default and override with ros params
        args.world_frame = rospy.get_param('~world_frame', args.world_frame)
        args.depth_cam_frame = rospy.get_param(
            '~input_topic_name', args.input_topic_name)
        args.depth_cam_frame = rospy.get_param(
            '~output_topic_name', args.output_topic_name)
        args.model = rospy.get_param('~model', args.model)
        args.resume = rospy.get_param('~resume', args.resume)
        self.args = args


if __name__ == '__main__':
    rospy.init_node("scene_completion")
    ri = ROSInfer()
    ri.start()
    rospy.spin()
