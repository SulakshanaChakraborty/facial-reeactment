from modules.config import create_parser
from modules.audio2keypoint import Audio2Keypoint
from utils.dsp import walk_filter
import argparse
import os
import pathlib
import json
from utils.vad import split_audio_with_vad
import numpy as np
from utils.dsp import load_audio
import timeit

if __name__ == '__main__':
    start = timeit.default_timer()
    parser = argparse.ArgumentParser()
    # TODO: rename parser
    parser = argparse.ArgumentParser(description='train model')
    parser = create_parser(parser)
   
    # parser.add_argument('--preprocessed_dir', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    args = parser.parse_args()
    # parser.add_argument('--debug', action='store_true')  # TODO: check parser for debug flag
 
    args.checkpoint = args.checkpoint_path

    # TODO: load keypoints from file (if testing with videos)
    # data/preprocessed must always exist at experiment dir with the lm_mean.npy and lm_std.npy
   
    args.mode = "predict"
    args.feature_type = 'deepspeech'
    args.mel_spec_concat = True
    test_files = walk_filter(f"{args.preprocessed_dir}/test", ".npz")

    keypoint_gan = Audio2Keypoint(args, debug=True)
    keypoint_gan.inference(test_files)
    stop = timeit.default_timer()

    print('Time: ', stop - start)  
   
