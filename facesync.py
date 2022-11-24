from modules.config import create_parser
from modules.audio2keypoint import Audio2Keypoint
from utils.dsp import load_audio
from utils.vad import split_audio_with_vad
from deltas.face_deltas import FaceDeltas
from deltas.head_deltas import HeadDeltas
from deltas.eyes_deltas import EyesDeltas
import argparse
import os
import pathlib
import numpy as np
import subprocess
import logging

# TODO: logging?


class FaceSync:
    def __init__(self, args, config, padded_pose_shape=None):
        self.padded_pose_shape = padded_pose_shape
        self.args = args
        self.args.mode = "inference"
        self.args.batch_size = 1
        self.model = Audio2Keypoint(self.args, seq_len=self.padded_pose_shape, debug=args.debug)
        self.frame_size = (760., 1200.)
        self.ref_landmarks = np.load("resources/leyla_landmarks.npy")  # TODO: bring back ref_landmarks arg
        self.config = config

    def process(self, audio_path, keypoints=None):
        signal = load_audio(audio_path, sr=self.args.sr)
        face_deltas = FaceDeltas(self.config["face_deltas"])
        head_deltas = HeadDeltas(self.config["head_deltas"])
        eyes_deltas = EyesDeltas(self.config["eyes_deltas"])
        norm_dist = np.max(self.ref_landmarks[:, 0]) - np.min(self.ref_landmarks[:, 0])
        if not keypoints:
            keypoints = np.array([])
            for segment in split_audio_with_vad(signal):
                segment_keypoints = self.model.predict_audio(segment, self.ref_landmarks, self.padded_pose_shape,
                                                             [0, 0])
                print("segment_keypoints: ",segment_keypoints.shape)
                face_deltas.extract_deltas(segment_keypoints, norm_dist, self.frame_size)
                head_deltas.extract_deltas(segment_keypoints, self.frame_size)
                if keypoints.shape[0] == 0:
                    keypoints = segment_keypoints
                else:
                    keypoints = np.concatenate([keypoints, segment_keypoints])
            eyes_deltas.extract_deltas(-np.array(head_deltas.pitch), -np.array(head_deltas.yaw))
        else:
            # TODO: test with keypoints arg
            face_deltas.extract_deltas(keypoints, norm_dist, self.frame_size)
            head_deltas.extract_deltas(keypoints, self.frame_size)
            eyes_deltas.extract_deltas(-np.array(head_deltas.pitch), -np.array(head_deltas.yaw))

        return face_deltas, head_deltas, eyes_deltas, keypoints


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = create_parser(parser)
    parser.add_argument('--audio_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--padded_pose_shape', type=int)
    args = parser.parse_args()

    facesync = FaceSync(args, padded_pose_shape=args.padded_pose_shape)
    print(facesync.process(args.audio_path))
