from modules.config import create_parser
import matplotlib
matplotlib.use('Agg')
import argparse
from modules.audio2keypoint import Audio2Keypoint
from utils.dsp import walk_filter
import tensorflow as tf
import random
import numpy as np

# set random seed
seed = 1337
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train model')
    parser = create_parser(parser)
    args = parser.parse_args()

    args.mode = 'train'
    args.feature_type = 'deepspeech'
    args.mel_spec_concat = False
    train_files = walk_filter(f"{args.preprocessed_dir}/train", ".npz")
    val_files = walk_filter(f"{args.preprocessed_dir}/val", ".npz")
    # put linda in front to use as display val files
    folders = list(set([f.split("/")[-2] for f in val_files]))
    if "linda" in folders:
        linda_files = [f for f in val_files if f.split("/")[-2] == "linda"]
        linda_files.extend([f for f in val_files if f.split("/")[-2] != "linda"])
        val_files = linda_files
    keypoint_gan = Audio2Keypoint(args, debug=True)
    keypoint_gan.train(train_files, val_files, args.batch_size, args.epochs)
