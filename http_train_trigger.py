#!/usr/bin/python3
import http.client
import json
import time
import rospy
from std_msgs.msg import String
from std_srvs.srv import Empty, EmptyResponse

class TrainTrigger:
    def __init__(self):
        rospy.init_node('http_train_trigger')
        self.start_trigger = rospy.Service('/cf_train/start', Empty, self.start)
        self.stop_trigger = rospy.ServiceProxy('/cf_train/stop', Empty)

        self.start_flag = False

        rospy.loginfo("http_train_trigger is initialized")

        while not rospy.is_shutdown():
            if self.start_flag:
                self.client()
                self.start_flag = False
                self.stop_trigger()
            time.sleep(1)

    def start(self, req):
        self.start_flag = True
        return EmptyResponse()

    def client(self):
        conn = http.client.HTTPConnection("localhost", 8080)
        conn.request("POST", "/cf_train", json.dumps({"name": "test"}))
        response = conn.getresponse().read()
        print(response)

if __name__ == '__main__':
    # rospy.init_node('http_train_trigger')
    trigger = TrainTrigger()
    # trigger.client()
    # while not rospy.is_shutdown():
    #     trigger.client()
    #     time.sleep(1)


