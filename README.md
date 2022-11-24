# Synthesising visual prosody in digital humans

## train the model

`python3 -W ignore train.py --output_path OUTPUT_PATH --preprocessed_dir INPUT_PATH`

In that OUTPUT_PATH there will be 3 folders: `checkpoints`, `tf_logs` and `val_samples`. Every 14 epochs a number of validation sample outputs are saved in `val_samples`, and at the end those are all aggregated into visualization files in `val_samples/epochs_plot`.

### Run inference

 `python run.py --audio_dir AUDIO_PATH --checkpoint_path CHEKPOINT_PATH`
