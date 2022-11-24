from modules.config import create_parser
from facesync import FaceSync
from utils.dsp import walk_filter
import argparse
import os
import pathlib
from dealer_client import DealerClient
from common.pose_plot_lib import save_facesync_debug_video
import json
import numpy as np

def save_debug_data(audio_path, keypoints, face_deltas, head_deltas, frame_size):
    plots_dir = f"{os.path.splitext(audio_path)[0]}"
    pathlib.Path(plots_dir).mkdir(exist_ok=True, parents=True)
    np.save(f"{plots_dir}/keypoints.npy", keypoints)
    save_facesync_debug_video(audio_path, keypoints, face_deltas, head_deltas, plots_dir, frame_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # TODO: rename parser
    parser = create_parser(parser)
    parser.add_argument('--audio_dir', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--config_path', type=str, default="deltas/config.json")
    parser.add_argument('--debug', action='store_true')  # TODO: check parser for debug flag
    parser.add_argument('--skip_done', action='store_true')
    parser.add_argument('--skip_facesync', action='store_true')
    parser.add_argument('--request_render', action='store_true')
    args = parser.parse_args()
    dc = DealerClient("http://159.138.99.24:24000")

    # TODO: load keypoints from file (if testing with videos)
    # data/preprocessed must always exist at experiment dir with the lm_mean.npy and lm_std.npy
    args.preprocessed_dir = f'{args.checkpoint_path.split("output")[0]}data/preprocessed'
    args.checkpoint = args.checkpoint_path
    args.feature_type = 'mel_spec'
    args.mel_spec_concat = False
    config = json.load(open(args.config_path))
  
    facesync = FaceSync(args, config, padded_pose_shape=576)
    print("in here?")
    audio_paths = walk_filter(args.audio_dir, ".wav")
    print("number of files: ",len(audio_paths))

    for audio_n, audio_path in enumerate(audio_paths):
        try:
            # skip facesyncing files that have been rendered
            if args.skip_done and args.request_render and os.path.exists(
                    f"{os.path.dirname(audio_path)}/renders/{os.path.basename(os.path.splitext(audio_path)[0])}_.mp4"):
                print(f"Skipping {audio_path} since it's already rendered")
                continue
            print(f"{audio_n + 1}/{len(audio_paths)} {audio_path}")

            
                # generate facesyncs
            if not args.skip_facesync:
                face_deltas, head_deltas, eyes_deltas, keypoints = facesync.process(audio_path,)
                # save to json
                result = {
                    'face': {
                        'eyebrow_l': face_deltas.eyebrow_l,
                        'eyebrow_r': face_deltas.eyebrow_r,
                        'eye_blinks': face_deltas.eye_blinks
                    },
                    'head': {
                        'roll': head_deltas.roll,
                        'pitch': head_deltas.pitch,
                        'yaw': head_deltas.yaw
                    },
                    'eyes': {
                        'x': eyes_deltas.x,
                        'y': eyes_deltas.y
                    }
                }
                json.dump(result, open(f"{os.path.splitext(audio_path)[0]}_facesync.json", 'w'), indent=4)
                json.dump(config, open(f"{os.path.dirname(audio_path)}/config.json", 'w'), indent=4)
                if args.debug:
                    save_debug_data(audio_path, keypoints, face_deltas, head_deltas, facesync.frame_size)

            if args.request_render:
                text_path = f"{os.path.splitext(audio_path)[0]}.txt"
                facesync_path = f"{os.path.splitext(audio_path)[0]}_facesync.json"
                dc.facesync_to_lipsync(text_path, audio_path, facesync_path)

        except Exception as e:
            print(f"Something went wrong: {e}")
            continue
