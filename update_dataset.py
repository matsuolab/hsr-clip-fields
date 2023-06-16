#!/usr/bin/env python3
import numpy as np
import time
import glob



class ReplaceDataset():
# "/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/init"のデータのうちオドメトリが/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/collect"の各データに対応するものとそれぞれ置き換える
    def __init__(self):
        start_time = time.time()
        # "/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/collect/"以下に含まれるデータの数を数える
        self.collect_num = len(glob.glob("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/collect/image*.npy"))
        # "/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/gpsr/"以下に含まれるデータの数を数える
        self.init_num = len(glob.glob("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/gpsr/image*.npy"))
        
        exchange_idices = []
        for idx in range(min(self.collect_num, int(self.init_num/5))):
            # "/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/collect/"のodomに最も近い"/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/gpsr/"のodomを探す

            # "/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/collect/"のodomを取得
            collect_odom = np.load("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/collect/odom"+str(idx).zfill(3)+".npy")

            # "/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/collect/"のodomと"/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/gpsr/"のodomの差を計算
            diff = [np.linalg.norm(collect_odom - np.load("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/gpsr/odom"+str(i).zfill(3)+".npy"), axis=1) for i in range(self.init_num)]
            # "/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/collect/"のodomに最も近い"/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/gpsr/"のodomのインデックスを取得
            # diff を値が小さい順に並び替える
            sorted_diff = np.argsort(diff)
            for exchange_idx in sorted_diff:
                if exchange_idx not in exchange_idices:
                    exchange_idices.append(exchange_idx)
                    break

        for idx in range(len(exchange_idices)):

            image = np.load("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/collect/image"+str(idx).zfill(3)+".npy")
            depth = np.load("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/collect/depth"+str(idx).zfill(3)+".npy")
            world = np.load("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/collect/world"+str(idx).zfill(3)+".npy")
            odom = np.load("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/collect/odom"+str(idx).zfill(3)+".npy")

            np.save("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/gpsr/image"+str(exchange_idices[idx]).zfill(3)+".npy", image)
            np.save("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/gpsr/depth"+str(exchange_idices[idx]).zfill(3)+".npy", depth)
            np.save("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/gpsr/world"+str(exchange_idices[idx]).zfill(3)+".npy", world)
            np.save("/root/catkin_ws/src/ros_docker/hsr_collection/scripts/hsr-clip-fields/data/gpsr/odom"+str(exchange_idices[idx]).zfill(3)+".npy", odom)

        print("time: ", time.time() - start_time)

if __name__ == "__main__":

    rd = ReplaceDataset()

