#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import tf
import tf2_ros
import std_msgs
import os
from geometry_msgs.msg import PoseStamped

INTRINSICS = {
    'realsense': [607.3814086914062, 0.0, 315.9123840332031, 0.0, 607.2514038085938, 233.77308654785156, 0.0, 0.0, 1.0],
    'xtion': [537.4933389299223, 0.0, 319.9746375212718, 0.0, 536.5961755975517, 244.54846607953, 0.0, 0.0, 1.0],
}

def get_info(tfl, use_hand, stamp=None, wait=True, target_frame='world'):
    hdr = std_msgs.msg.Header()
    hdr.stamp = rospy.Time(0) if stamp is None else stamp

    if use_hand:
        hdr.frame_id = 'hand_camera_color_optical_frame'#'realsense_link'
        K = INTRINSICS['realsense']
    else:
        hdr.frame_id = 'xtion_link'
        K = INTRINSICS['xtion']

    if wait:
        tfl.waitForTransform('/' + target_frame, '/' + hdr.frame_id, hdr.stamp, rospy.Duration(3.0))

    mat = tfl.asMatrix('/' + target_frame, hdr)

    return K, mat

def depth2cam(dx, dy, dz, K):
    cx = (dx - K[2]) * dz / K[0]
    cy = (dy - K[5]) * dz / K[4]

    return cx, cy, dz


def depth2world(d_img, use_hand, tfl, stamp=None, wait=True, A=None):
    K, mat = get_info(tfl, use_hand, stamp=stamp, wait=wait)

    if A is not None:
        mat = A.dot(mat)

    cz = d_img.flatten()
    ix, iy = np.meshgrid(np.arange(640), np.arange(480))
    ix, iy = ix.flatten(), iy.flatten()

    cx, cy, _ = depth2cam(ix, iy, cz, K)

    X = np.column_stack([cx, cy, cz, np.ones_like(cx)])
    X = X.dot(mat.T)

    wx, wy, wz = np.split(X[:, :3], 3, 1)
    wx, wy, wz = wx.reshape(480, 640), wy.reshape(480, 640), wz.reshape(480, 640)
    w = np.dstack([wx, wy, wz])

    return w

class DataCollector:

    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/relay/head/rgb", Image, self.image_callback)
        self.depth_sub = rospy.Subscriber("/relay/head/depth", Image, self.depth_callback)
        self.odom_topic_sub=rospy.Subscriber('/global_pose',PoseStamped,self.Odom_scan)
        self.image = None
        self.depth = None
        self.tfl = tf.TransformListener(cache_time=rospy.Duration(100))

    def image_callback(self, data):
        cv_array = self.bridge.imgmsg_to_cv2(data, "bgr8")
        cv_array = cv2.cvtColor(cv_array, cv2.COLOR_BGR2RGB)
        self.image = cv_array

    def depth_callback(self, data):
        cv_array = self.bridge.imgmsg_to_cv2(data, "passthrough")
        self.depth = cv_array
        self.depth_stamp = data.header.stamp

    def Odom_scan(self,data):
        self.pose_stamped= [data.pose.position.x,data.pose.position.y,data.pose.position.z]

    def save_image(self, num, image, depth, world):
        image = image.astype(np.uint8)
        depth = depth.astype(np.uint16)
        world = world.astype(np.float32)
        odom = np.array(self.pose_stamped)
        num = "000"+str(num)
        num = num[-3:]
        np.save("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/data/"+ self.tag + "/image" + num +".npy", image)
        np.save("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/data/"+ self.tag + "/depth" + num +".npy", depth)
        np.save("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/data/"+ self.tag + "/world" + num +".npy", world)
        np.save("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/data/"+ self.tag + "/odom" + num +".npy", odom)


    def main(self):
        # ctrlC と入力するまでデータを10hzで取り続ける
        self.tag = input("タグを入力してください!")
        if os.path.isdir("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/" + self.tag):
            print("そのタグは既に存在します")
            return
        os.mkdir("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/" + self.tag)
            
        while True:
            key = input("sを押すとデータ取得を開始します, qを押すと終了します")
            if key == "s":
                break
            # cnrlC で終了
            elif key == "q":
                return

        num = 0
        while not rospy.is_shutdown():
            try:
                if self.image is None or self.depth is None:
                    print("image or depth is None")
                    continue
                while True:
                    try:
                        world = depth2world(self.depth / 1000.0, False, self.tfl, stamp=self.depth_stamp)
                        self.save_image(num, self.image, self.depth, world)
                        num += 1
                        break
                    except tf2_ros.TransformException as e:
                        print(e)
                rospy.sleep(1)
            except KeyboardInterrupt:
                print("終了します")
                return
        print("finish")
    



if __name__ == '__main__':
    rospy.init_node('auto_data_collector', anonymous=True)
    dc = DataCollector()
    dc.main()

