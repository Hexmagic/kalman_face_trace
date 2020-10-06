# Targets:
# A. 纯track
# B. Track + Detection
#
# 问：1. 如何表示物体？: list() 每个元素代表一个物体
#    2. 如何更新运动状态： update (update_with_detection)/update_without_detection
import numpy as np
import cv2
import os
import sys
from config import MAX_NUM_MISSING_PERMISSION, MAX_PIXELS_DIST_BETWEEN_PREDICTED_AND_DETECTED, BBXES_IDENTICAL_IOS_TRHESHOLD
import util

from scipy.optimize import linear_sum_assignment

from instance import Instance


class MultipleObjectController(object):
    def __init__(self):
        # instances: 关于instance类的list
        self.instances = []  # [instance1, instance2, ..., instance_n]
        self.prob_instances = []

    def update(self, detections):
        # update with detection
        # 用detection的结果结合tracking(prediction)的结果，更新物体运动状态
        self.assign_detections_to_tracks(detections)

    def update_without_detection(self, frame):
        # 只prediction，没有detection辅助correction
        # step 1: update bbx by prediction
        for instance in self.instances:  # 辅助
            [a, b, c, d] = instance.get_predicted_bbox()
            #tag = instance.get_latest_record()[0]
            # draw something...
            #[a, b, c, d] = map(int,util.centroid_to_bbx(center))
            cv2.rectangle(frame, (a, b), (c, d), (0, 255, 0), 2)
            instance.num_misses += 1  # 此行最重要
        # step 2: remove dead bboxes

        self.remove_dead_instances()

    ###########
    def assign_detections_to_tracks(self, detections):
        # A. no instances now: initial—>将detection的结果加入instances
        # B. we have instances now: (prediction/track<—>detection matching)
        #    1. predict/track—> kalman filter
        #    2. 找到detection和prediction相互的匹配
        #       2.a Munkres Algorithm
        #           det:   b    e f    //       b->b       e->f f->f
        #           pre: a b c    f    // a->b  b->b c->b       f->f
        #           matched: b, f
        #       2.b bi-matching: 以predict/track为基准，先匹配det；再由det为基准，匹配pre/track
        #    3. 对于正确匹配的物体，进行更新：correction
        #    4. 处理未正确匹配的物体
        #       4.1 未检测到的物体：a.即时移除(不建议) b.继续更新(单纯pred/track，不考虑detection)
        #                        c.长久未匹配，移除
        #       4.2 检测到有新物体：a. 即时加入 b.加入track(试用期)，几帧后(通过适用期后)，加入instances

        if len(self.instances) == 0:
            for det in detections:
                # det: 检测结果
                # det: {'tag': [bbox_left, bbox_right, bbox_top, bbox_bottom]}
                instance = Instance()
                tag = list(det.keys())[0]
                bbox = det[tag]
                instance.add_to_track(tag, bbox)  # 辅助
                self.instances.append(instance)
        # B.
        # B.1
        # 算距离：检测框与预测框之间的距离
        # costs[i, j]: 每一个预测框与检测框之间的距离
        costs = np.zeros(shape=(len(self.instances), len(detections)))
        matched = 0
        for i, instance in enumerate(self.instances):
            # Here, by using Kalman Filter, we predict an bbx for each instance
            predicted_bbx = instance.get_predicted_bbox()
            for j, det in enumerate(detections):
                detected_bbx = list(det.values())[0]
                dist = util.dist_btwn_bbx_centroids(
                    util.bbx_to_lrtb(predicted_bbx),
                    util.bbx_to_lrtb(detected_bbx))
                max_dist = MAX_PIXELS_DIST_BETWEEN_PREDICTED_AND_DETECTED
                if dist > max_dist:
                    dist = 1000  # sys.maxsize
                costs[i, j] = dist
        # set all tracked instances as unassigned
        for instance in self.instances:
            instance.has_match = False

        # B.2
        # 利用cost矩阵寻找匹配
        # Munkres Algorithm
        # Instructions and C# version can be found here: http://csclab.murraystate.edu/~bob.pilgrim/445/munkres.html
        # while in Python we can solve the problem by using method imported as below.
        # Descriptions for Python version can be found in
        # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
        assigned_instances, assigned_detections = linear_sum_assignment(costs)

        # B.3
        # 更新assigned bbox
        assigned_detection_id = []
        for idx, instance_id in enumerate(assigned_instances):
            detection_id = assigned_detections[idx]
            # if assignment for this instance and detection is sys.maxsize, discard it
            if costs[instance_id, detection_id] != 1000:  # sys.maxsize:
                assigned_detection_id.append(detection_id)
                self.instances[instance_id].has_match = True
                matched +=1
                self.instances[instance_id].correct_track(
                    detections[detection_id]['bbox'])
                self.instances[instance_id].num_misses = 0
        print(f'matched {matched}')
        # B.4
        # keep track of how many times a track has gone unassigned
        for instance in self.instances:
            if instance.has_match is False:
                instance.num_misses += 1
        # The function shown below can only remove those instances which has already been
        # added to tracks but CAN NOT remove detected bbx which has a huge IOU with
        # existed tracks. So we need another remove function to dual with that
        self.remove_dead_instances()

        # get unassigned detection ids
        unassigned_detection_id = list(
            set(range(0, len(detections))) - set(assigned_detection_id))
        for idx in range(0, len(detections)):
            if idx in unassigned_detection_id:
                # det: {'tag' : [bbx_left, bbx_right, bbx_up, bbx_bottom]}
                tag = list(detections[idx].keys())[0]
                bbx = detections[idx][tag]
                # then we need to confirm whether the detection is a good one
                if self.is_good_detection(bbx):
                    instance = Instance()
                    instance.add_to_track(tag, bbx)
                    self.instances.append(instance)

    def is_good_detection(self, bbx):
        #
        for instance in self.instances:
            if util.check_bbxes_identical_by_ios(
                    instance.get_latest_bbx(), bbx,
                    BBXES_IDENTICAL_IOS_TRHESHOLD):
                return False
        return True

    def remove_dead_instances(self):
        instances = []
        for instance in self.instances:
            if instance.num_misses < instance.max_misses:
                instances.append(instance)
        prob_instances = []
        for p in self.prob_instances:
            if p.num_misses < p.max_misses:
                prob_instances.append(p)
        self.instances = instances
        self.prob_instances = prob_instances

    def probation_to_instances(self):
        for idx, p in enumerate(self.prob_instances):
            if p.prob_times > p.max_probation:
                self.instances.append(p)
                self.prob_instances.pop(idx)
