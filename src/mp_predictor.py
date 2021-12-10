# -*- coding: utf-8 -*-
# Detectron2 & Mediapipe implementation

import mediapipe as mp
import cv2.cv2 as cv2
import os

from src.visualization import visualize_keypoints
from src.geometry import get_central_points

_KEYPOINT_THRESHOLD = .5


def get_updated_keypoints(keypoint_dict):
    points_to_connect = get_central_points(keypoint_dict)
    keypoint_dict.update(points_to_connect)
    return keypoint_dict


class MpipePredictor(mp.solutions.pose.Pose):
    def __init__(self, detection_thr, tracking_thr=.99, path_to_video=None, static=False, instance=0):
        super().__init__(static_image_mode=static, min_detection_confidence=detection_thr,
                         min_tracking_confidence=tracking_thr)
        self.instance = instance
        self.path_to_video = path_to_video
        if self.path_to_video is not None:
            self.video = cv2.VideoCapture(self.path_to_video)
            self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.scale = max(self.height, self.width) / 850
            self.frames_per_second = self.video.get(cv2.CAP_PROP_FPS)
            self.num_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.basename = os.path.basename(self.path_to_video)
        self.keypoints = {}
        self.tracking = {}

    def get_keypoints(self, img, get3d=False):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        results = self.process(img)
        if results is None:
            return
        idxs = mp.solutions.pose.PoseLandmark
        self.keypoints = {'nose': results.pose_landmarks.landmark[idxs.NOSE],
                          'left_eye': results.pose_landmarks.landmark[idxs.LEFT_EYE],
                          'right_eye': results.pose_landmarks.landmark[idxs.RIGHT_EYE],
                          'left_ear': results.pose_landmarks.landmark[idxs.LEFT_EAR],
                          'right_ear': results.pose_landmarks.landmark[idxs.RIGHT_EAR],
                          'left_shoulder': results.pose_landmarks.landmark[idxs.LEFT_SHOULDER],
                          'right_shoulder': results.pose_landmarks.landmark[idxs.RIGHT_SHOULDER],
                          'left_elbow': results.pose_landmarks.landmark[idxs.LEFT_ELBOW],
                          'right_elbow': results.pose_landmarks.landmark[idxs.RIGHT_ELBOW],
                          'left_wrist': results.pose_landmarks.landmark[idxs.LEFT_WRIST],
                          'right_wrist': results.pose_landmarks.landmark[idxs.RIGHT_WRIST],
                          'left_hip': results.pose_landmarks.landmark[idxs.LEFT_HIP],
                          'right_hip': results.pose_landmarks.landmark[idxs.RIGHT_HIP],
                          'left_knee': results.pose_landmarks.landmark[idxs.LEFT_KNEE],
                          'right_knee': results.pose_landmarks.landmark[idxs.RIGHT_KNEE],
                          'left_ankle': results.pose_landmarks.landmark[idxs.LEFT_ANKLE],
                          'right_ankle': results.pose_landmarks.landmark[idxs.RIGHT_ANKLE]}
        keypoints_dict = {}
        for name, obj in self.keypoints.items():
            keypoints_dict.update({name: self._get_coords(obj, img.shape, get3d)})
        return keypoints_dict

    def _get_coords(self, keypoint, img_shape=None, with_z=True):
        """get coords for keypoint"""
        x, y, z, vis = keypoint.x, keypoint.y, keypoint.z, keypoint.visibility

        if img_shape is not None:
            image_height, image_width, _ = img_shape
            image_depth = (image_height + image_width) / 2
            x, y, z = int(x * image_width), int(y * image_height), int(z * image_depth)

        if not with_z:
            return x, y, vis

        return x, y, z, vis

    def _frame_from_video(self, video):
        # while video.isOpened():
        f = 0
        while f < self.num_frames:
            success, frame = video.read()
            if success:
                yield frame
                f += 1
            else:
                break


if __name__ == "__main__":
    # test file
    im = cv2.imread("testimages/test.jpg")

    # example of mediapipe inference
    predictor = MpipePredictor(detection_thr=.8, tracking_thr=.9)
    kps = predictor.get_keypoints(im)