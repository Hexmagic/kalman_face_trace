import copy
import os

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torch.tensor import Tensor
from torchvision.transforms import ToTensor

from config import NUM_JUMP_FRAMES
from data import cfg
from detector import Detector  # Face Detector
from layers.functions.prior_box import PriorBox
#from acceptor import Acceptor
from models.faceboxes import FaceBoxes
from multiple_object_controller import MultipleObjectController
from utils.box_utils import decode
from utils.nms_wrapper import nms


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(
            pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def detect(detector, frame):
    img = np.float32(frame)
    img -= (104, 117, 123)
    tensor = ToTensor()(img)

    im_height, im_width, _ = frame.shape
    var = Variable(tensor).to('cpu').unsqueeze(0)
    loc, conf = detector(var)
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    img = frame
    scale = torch.Tensor(
        [img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    priors = priors.to('cpu')
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / 1
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    # ignore low scores
    inds = np.where(scores > 0.5)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:5000]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
        np.float32, copy=False)
    #keep = py_cpu_nms(dets, args.nms_threshold)
    keep = nms(dets, 0.5, force_cpu=True)
    dets = dets[keep, :]

    # keep top-K faster NMS
    dets = dets[:750, :]
    rtn = []
    for ele in dets.tolist():
        rtn.append({'bbox': list(map(int, ele))[:-1]})
    return rtn


def run():
    # Step 1: Initialization
    # 视频： video_helper.py
    # 检测： detector.py
    # 结果接收：acceptor.py
    # 参数配置: config.py
    # 总控： multiple_object_controller.py
    #detector = Detector()
    #acceptor = Acceptor()
    #video_helper = VideoHelper()
    # initialize detector
    net = FaceBoxes(phase='test', size=None, num_classes=2)
    net = load_model(net, 'weights/FaceBoxes.pth', True)
    net.eval()
    detector = net
    object_controller = MultipleObjectController()

    # step 2: 总体流程：main loop
    # A: 对物体，每帧检测，不要跟踪 （可以要平滑）
    # B: 对物体，要跟踪： a. 此帧有检测 (+observation correction)
    #                  b. 此帧无检测（只跟踪，pure predicton）
    cur_frame_counter = 0
    detection_loop_counter = 0
    cap = cv2.VideoCapture('demo.mp4')
    import time
    while True:
        # 0. get frame
        ret, ori_frame = cap.read()
        frame = copy.copy(ori_frame)
        #frame = cv2.imread('test.jpg')
        # 1.1 每帧都检测
        if not NUM_JUMP_FRAMES:
            detects = detector.detect(frame)
            object_controller.update(detects)
        else:
            # 1.2 隔帧检测
            # 1.2.1 此帧有检测
            if detection_loop_counter % NUM_JUMP_FRAMES == 0:
                #detection_loop_counter = 0
                detects = detect(detector, frame)
                for det in detects:
                    x,y,r,b = det['bbox']
                    cv2.rectangle(frame,(x,y),(r,b),(0,255,0),2)
                object_controller.update(detects)  # 核心
            # 1.2.2 此帧无检测
            else:
                object_controller.update_without_detection(frame)  # 核心
        cv2.imshow('Trace', frame)
        #cv2.imshow('Origin',ori_frame)
        # deal with acceptor
        detection_loop_counter += 1
        cv2.imwrite(f'output/{detection_loop_counter}.jpg',frame)
        #time.sleep(0.05)
        k = cv2.waitKey(5) & 0xff
        if k == 27:  # 'esc' key has been pressed, exit program.
            break
        if k == 112:  # 'p' has been pressed. this will pause/resume the code.
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
