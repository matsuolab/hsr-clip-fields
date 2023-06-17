#!/usr/bin/python3
import http.client
import json
import time
import rospy
from std_msgs.msg import String
from hsr_collection.srv import CFPred, CFPredResponse


class PredTrigger:
    def __init__(self):
        rospy.init_node('http_pred_trigger')

        self.pred_srv = rospy.Service('/cf_pred', CFPred, self.pred)

        rospy.loginfo("http_pred_trigger is initialized")

    def pred(self, req):
        self.query = req.query
        return CFPredResponse(self.client())

    def client(self):
        conn = http.client.HTTPConnection("localhost", 8000)
        conn.request("POST", "/cf_pred", json.dumps({"query": self.query}))
        response = conn.getresponse().read()
        # python dictに変換
        response = json.loads(response)
        print(response)
        coordinates = response[0][self.query]
        print(coordinates)
        return coordinates

if __name__ == '__main__':
    # rospy.init_node('http_train_trigger')
    trigger = PredTrigger()
    rospy.spin()
    # while not rospy.is_shutdown():
    #     trigger.client()
    #     time.sleep(1)


