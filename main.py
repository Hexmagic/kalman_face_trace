import cv2

import os
import cv2
import numpy as np

from config import NUM_JUMP_FRAMES
from detector import Detector  # Face Detector
from multiple_object_controller import MultipleObjectController
#from acceptor import Acceptor


def run():
    # Step 1: Initialization
    # 视频： video_helper.py
    # 检测： detector.py
    # 结果接收：acceptor.py
    # 参数配置: config.py
    # 总控： multiple_object_controller.py
    detector = Detector()
    #acceptor = Acceptor()
    #video_helper = VideoHelper()
    object_controller = MultipleObjectController()

    # step 2: 总体流程：main loop
    # A: 对物体，每帧检测，不要跟踪 （可以要平滑）
    # B: 对物体，要跟踪： a. 此帧有检测 (+observation correction)
    #                  b. 此帧无检测（只跟踪，pure predicton）
    cur_frame_counter = 0
    detection_loop_counter = 0
    cap = cv2.VideoCapture('bug.mp4')
    i = 0
    while True:
        # 0. get frame
        i += 1
        ret, frame = cap.read()
        if i < 20:
            continue

        # 1.1 每帧都检测
        if not NUM_JUMP_FRAMES:
            detects = detector.detect(frame)
            object_controller.update(detects)
        else:
            # 1.2 隔帧检测
            # 1.2.1 此帧有检测
            if detection_loop_counter % NUM_JUMP_FRAMES == 0:
                detection_loop_counter = 0
                detects = detector.detect(frame)
                object_controller.update(detects)  # 核心
            # 1.2.2 此帧无检测
            else:
                object_controller.update_without_detection()  # 核心

        # deal with acceptor
        # ask acceptor do something
        cur_frame_counter += 1
        detection_loop_counter += 1
        k = cv2.waitKey(50) & 0xff
        if k == 27:  # 'esc' key has been pressed, exit program.
            break
        if k == 112:  # 'p' has been pressed. this will pause/resume the code.
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
