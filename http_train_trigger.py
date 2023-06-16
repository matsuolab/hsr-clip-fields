#!/usr/bin/python3
import http.client
import json
import time
# import rospy
from std_msgs.msg import String

class Trigger:
    def client(self):
        conn = http.client.HTTPConnection("localhost", 8080)
        conn.request("POST", "/cf_server", json.dumps({"name": "test"}))
        response = conn.getresponse().read()
        print(response)

if __name__ == '__main__':
    # rospy.init_node('http_train_trigger')
    trigger = Trigger()
    trigger.client()
    # while not rospy.is_shutdown():
    #     trigger.client()
    #     time.sleep(1)


