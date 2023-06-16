#!/usr/bin/python3
import http.client
import json
import time
import rospy
from std_msgs.msg import String
from hsr_collection.srv import CFPred, CFPredResponse


class PredTrigger:
    def client(self):
        query = "snack"
        conn = http.client.HTTPConnection("localhost", 8000)
        conn.request("POST", "/cf_pred", json.dumps({"query": query}))
        response = conn.getresponse().read()
        # python dictに変換
        response = json.loads(response)
        print(response)
        coordinates = response[0][query]
        print(coordinates)

if __name__ == '__main__':
    # rospy.init_node('http_train_trigger')
    trigger = PredTrigger()
    trigger.client()
    # while not rospy.is_shutdown():
    #     trigger.client()
    #     time.sleep(1)


