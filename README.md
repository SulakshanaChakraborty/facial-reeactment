### Facesync

#### Preparing the data
The goal is to extract a set of .npz files with validated pose and audio signals from raw data.

Starting from raw data located at /media/ssd_3/n_project/data/0_raw/arabic/Week#2 which may comprise .mp4/.mov files @ 30 fps you can follow these steps:

1. Convert the videos to .mp4 @ 25 fps by running:

`python3 convert_videos.py /media/ssd_3/n_project/data/0_raw/arabic/Week#2 /media/ssd_3/n_project/data/1_preprocessed/arabic_week2`

and

`python3 convert_videos.py /media/ssd_3/n_project/data/0_raw/arabic/Week#2 /media/ssd_3/n_project/data/1_preprocessed/arabic_week2 --ext MOV`

2. Then extract the keypoints by running:

`python extract_landmarks.py --video_dir /media/ssd_3/n_project/data/1_preprocessed/arabic_week2 --skip_done`

For the extract_landmarks.py function you can use the default python3 from the alienware which contains the necessary packages for face_alignment.

3. Copy the 16kHz .wav audio files are from /media/ssd_3/n_project/data/0_raw/arabic/Week#2 into /media/ssd_3/n_project/data/1_preprocessed/arabic_week2 if they exist, otherwise extract them with an ffmpeg script.

4. Make a new experiment directory at /media/ssd_2/facesync/experiments eg. 202111161237-exp_3 (and make a `data/preprocessed` dir inside of it) and run the data extraction script:

`python3 extract_data.py --root /media/ssd_3/n_project/data/1_preprocessed/arabic_week2 --output_dir /media/ssd_2/facesync/experiments/202111161237-exp_3/data/preprocessed --n_jobs 4 --augment_factor 2 --train_percent 0.9`

--augment_factor performs sliding window augmentation to obtain double the amount of data. Note extract_data.py performs validation and will discard wrong samples.

(5). For an example of aggregating multiple datasets view /media/ssd_2/facesync/experiments/202111161237-exp_3/data/preprocessed. Note that when aggregating datasets you must recompute the training data statistics used in normalisation by running:

`python3 compute_stats.py --data_dir /media/ssd_2/facesync/experiments/202111161237-exp_3/data/preprocessed/train`

And copying over the `lm_mean.npy` and `lm_std.npy` files to /media/ssd_2/facesync/experiments/202111161237-exp_3/data/preprocessed. The compute_stats.py is located in old_files.

#### Launching a training run
Make an `output` dir in the experiment folder and run:

`python3 -W ignore train.py --output_path /media/ssd_2/facesync/experiments/202111161237-exp_3/output --preprocessed_dir /media/ssd_2/facesync/experiments/202111161237-exp_3/data/preprocessed`

In that dir there will be 3 folders: `checkpoints`, `tf_logs` and `val_samples`. Every 14 epochs a number of validation sample outputs are saved in `val_samples`, and at the end those are all aggregated into visualization files in `val_samples/epochs_plot`.


#### Running inference
Let's say we have a compressed folder game_cutscenes-20211206T124359Z-001.zip with audio and text files. We decompress it into a folder in `experiments/202112061344-game_cutscenes` and then run:

 `python run.py --audio_dir experiments/202112061344-game_cutscenes --checkpoint_path experiments/202111161237-exp_3/output/checkpoints/ckpt-step-299000.ckp --request_render`

Typically I run this on my local machine as it makes it easier to debug. So the above command works just scp the desired checkpoint files and statistics files and save them as:
experiments/202111161237-exp_3/output/checkpoints/ckpt-step-299000.ckp.data-00000-of-00001
experiments/202111161237-exp_3/output/checkpoints/ckpt-step-299000.ckp.index
experiments/202111161237-exp_3/output/checkpoints/ckpt-step-299000.ckp.meta
experiments/202111161237-exp_3/data/preprocessed/lm_mean.npy
experiments/202111161237-exp_3/data/preprocessed/lm_std.npy

For debugging purposes there's --debug, it saves a visualization of the inference in the directory of the files. This adds to the runtime though, I recommend just using it when debugging a few files at a time.

When getting the facesyncs for demo data I usually do that from David. Let's say Oscar has prepared some TTS samples in /media/ssd_sata3/readspeaker_tts/audio/ar/Arabic_1208_demo_tts/yasmin/110, then from /media/ssd_pcie1/luca/facesync you can run:

`python run.py --audio_dir /media/ssd_sata3/readspeaker_tts/audio/ar/Arabic_1208_demo_tts/yasmin/110 --checkpoint_path /media/ssd_pcie0/luca/facesync/experiments/202111161237-exp_3/output/checkpoints/ckpt-step-299000.ckp`

#### Running inference on facesync-engine
Having the facesync-engine up & running at port 8080 (default --endpoint), you can get results by running:

`python run_from_engine.py --data_dir experiments/202111221149-1_to_1_with_engine`

Where experiments/202111221149-1_to_1_with_engine contains all the audio files.