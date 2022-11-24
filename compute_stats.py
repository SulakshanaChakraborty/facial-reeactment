import argparse
import os
# import concurrent.futures as ft
import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf

# tf.disable_v2_behavior()
import functools
from tensorflow.python.ops.signal import window_ops
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# from utils.dsp import  walk_filter
# tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)

# def compute_pitch_stats(data_files):
#     n = len(data_files)
#     print("len of data files: ",n)
    
#     min = np.inf
#     max = -np.inf

#     for file_path in data_files:
#         val = np.load(file_path)
        
#         new_min = np.min(val)
        

#         new_max = np.max(val)


#         if new_min < min: min = new_min
#         if new_max > max: max = new_max

#     print("pitch extraction completed!")
#     return min,max

# def compute_mel_stats(data_files,n_jobs):
#     n = len(data_files)
#     print("len of data files: ",n)
    
#     min = np.inf
#     max = -np.inf

#     # with ft.ProcessPoolExecutor(max_workers=n_jobs) as worker:
#     #     for file_path in data_files:
#     #         audio = np.load(file_path)['audio']
#     #         f = worker.submit(mel_spec_from_audio,audio)
#     #         val = f.result()
#     #         new_min = np.min(val)
#     #         print(new_min)
#     #         new_max = np.max(val)
   
#     #         if new_min < min: min = new_min
#     #         if new_max > max: max = new_max


#     for file_path in data_files:
#             audio = np.load(file_path)['audio']
#             val = mel_spec_from_audio(audio)
#             new_min = np.min(val)
#             # print(new_min)
#             # print("min curr: ",new_min)
#             new_max = np.max(val)
#             # print("max curr: ",new_max)

#             if new_min < min:
#                 min = new_min
#             if new_max > max: 
#                 max = new_max
            

#     print("mel stats completed!")
#     return min,max 

def compute_mel_stats(file_names):
    n = len(file_names)
    running_sum = np.sum((64))
    running_sum_2 = np.sum((64))
    audio = tf.placeholder(tf.float32,shape =[1,40960])
    # mel_arr = np.zeros((n,254,64))
    # print(file_names)
    tf_out = mel_spec_from_audio(audio)
    with tf.Session() as sess:
        for idx, file in enumerate(file_names):
            
            print("extract mel file id: ",idx)
            zip_file = np.load(file)
            mel_spec = sess.run(tf_out,feed_dict={audio:zip_file['audio']})
            
            # mel_spec_from_audio(zip_file['audio'])
            running_sum += np.sum(mel_spec).squeeze()
            running_sum_2 += np.sum(mel_spec**2).squeeze()
            # mel_arr[idx] = mel_spec
    # _,m,l = mel_spec.shape
    # print(mel_spec.shape)
    num_elem = n*254*64
    # print(mel_arr.shape)
    # print("num_elem: ",num_elem)
    mean_tot = running_sum/num_elem

    # sum_np=np.sum(mel_arr)
    # print("the sum: ",sum_np)
    # print("length: ",n)

    # print("running_sum: ",running_sum)
    # print("running_sum_2: ",running_sum_2)
    

    std_tot = np.sqrt((running_sum_2 /num_elem) - (mean_tot **2)) 
    

    #test
    # numpy_mean = np.mean(mel_arr)
    # numpy_std = np.std(mel_arr)
    print("mean_tot: ",mean_tot)
    # print("numpy_mean: ",numpy_mean)
    print("std_tot: ",std_tot)
    # print("numpy_std: ",numpy_std)

    # print(f"is numpy and manual stats equal? mean: {(numpy_mean == mean_tot).all()},{(numpy_std == std_tot).all()}" )
    return mean_tot,std_tot

def mel_spec_from_audio(res_audio):
    # x_audio = res_audio.astype('float32')
    # x_audio = tf.constant(x_audio)
    stft = tf.signal.stft(
        res_audio,
        400, # frame_length , the window length
        160, # originally 160, reduces the upsampling discrepancy in encoder # frame_step , hop size
        fft_length= 512,
        window_fn= functools.partial(window_ops.hann_window, periodic=True),
        pad_end=False,
        name=None
    )
    stft = tf.abs(stft)
    # Returns a matrix to warp linear scale spectrograms to the mel scale
    mel_spect_input = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=64,
        num_spectrogram_bins=tf.shape(stft)[2], #257
        sample_rate=16000,
        lower_edge_hertz=125.0,
        upper_edge_hertz=7500.0,
        dtype=tf.float32,
        name=None
    )
    input_data = tf.tensordot(stft, mel_spect_input, 1)
    input_data = tf.math.log(input_data + 1e-6)

    # res = input_data.eval(session=tf.compat.v1.Session())
    # res = input_data.numpy()
    return input_data

def walk_filter(input_dir, file_extension=None):
    files = []
    #print("input_dir: ",input_dir)
    #print("file_extension: ",file_extension)
    for r, _, fs in os.walk(input_dir, followlinks=True):
        if file_extension:
            files.extend([os.path.join(r, f) for f in fs if os.path.splitext(f)[-1] == file_extension])
        else:
            files.extend([os.path.join(r, f) for f in fs])
    #print("file paths fetched: ",files)
    return files
    
def compute_pitch_stats(audio_data_files):

    n = len(audio_data_files)
    pitch_arr = np.zeros((n,254))
    for idx, data_path in enumerate (audio_data_files):
        file_name = os.path.splitext(os.path.basename(data_path))[0]
        features_file = file_name+"_features.npz"
        feature_path = "/media/ssd/sulakshana/facesync_sulak/data/arabic/features_extracted/"+features_file
        # path_local = r"C:\Users\Sulakshana\Desktop\STUDY\Course Work\thesis project\remote\input data\pitch_extracted"
        # pitch_path =   os.path.join(path_local,pitch_file)
        # print(pitch_path)
        feature_file = np.load(feature_path)
        pitch_arr[idx] = feature_file['pitch']
    #pitch_arr = np.log(pitch_arr +1e-6)
    mean = np.mean(pitch_arr) 
    std = np.std(pitch_arr)

    return mean,std

def compute_egmaps_stats(audio_data_files):

    n = len(audio_data_files)
    egmaps_arr = np.zeros((n,254,25))
    for idx, data_path in enumerate (audio_data_files):
        file_name = os.path.splitext(os.path.basename(data_path))[0]
        features_file = file_name+"_features.npz"
        feature_path = "/media/ssd/sulakshana/facesync_sulak/data/features_extracted/"+features_file
        # path_local = r"C:\Users\Sulakshana\Desktop\STUDY\Course Work\thesis project\remote\input data\pitch_extracted"
        # feature_path =   os.path.join(path_local,features_file)
        # print(pitch_path)
        feature_file = np.load(feature_path)
        arr = feature_file['egmaps']
        egmaps_arr[idx] = arr
    #pitch_arr = np.log(pitch_arr +1e-6)
    egmaps_arr=egmaps_arr.reshape(-1,25)
    mean = np.mean(egmaps_arr,axis =0) 
    std = np.std(egmaps_arr,axis=0)

    return mean,std

if __name__ == '__main__':
    #print("is the code even runnig?")
    parser = argparse.ArgumentParser()

    #audio_dir  = "/media/ssd_2/facesync/experiments/202201061209-exp_1/data/preprocessed/train"
    audio_dir  = "/home/sulakshana/facesync_sulak/data/preprocessed/train"
    # output_dir = "/home/sulakshana/facesync_sulak/data/pitch_extracted"

    # audio_dir  = r"C:\Users\Sulakshana\Desktop\STUDY\Course Work\thesis project\remote\input data\pitch_extracted"
    # pitch_dir  = r"C:\Users\Sulakshana\Desktop\STUDY\Course Work\thesis project\remote\input data\pitch_extracted"
    # output_dir = r"C:\Users\Sulakshana\Desktop\STUDY\Course Work\thesis project\remote\check"

    parser.add_argument('--audio_dir', type = str, default = audio_dir, help='path to the input data')
    # parser.add_argument('--pitch_dir', type = str, default = pitch_dir, help='path to the input data')
    # parser.add_argument('--output_dir', type =str, default = output_dir, help='path to output dir')
    parser.add_argument('--n_jobs', type=int, default=8, help='number of parallel processes')
    args = parser.parse_args()
    # file_extension = ".npz"
    # audio_data_files = walk_filter(args.audio_dir,file_extension)
    file_extension = ".npz"
    data_files = walk_filter(args.audio_dir,file_extension)
    print("number of file: ",len(data_files))

    # pre-process
    
    # standerdise
    stats_dict = {}
    stats_dict['pitch']={}
    stats_dict['mel']={}
    pitch = False
    if pitch:
        mean,std = compute_pitch_stats(data_files)
        np.savez_compressed('arabic_pitch_stats', a=mean, b=std)
        print("pitch stats: ",(mean,std))

    # print("for pitch: ", (mean,std))
    # stats_dict['pitch']['mean']=str(mean)
    # stats_dict['pitch']['std']=str(std)
    gemaps = True
    if gemaps:
        mean,std = compute_egmaps_stats(data_files)
        np.savez_compressed('arabic_egmaps_stats', a=mean, b=std)
        print("egmaps stats: ",(mean.shape,std.shape))


    mel= False
    if mel:
        mean,std = compute_mel_stats(data_files)
        print("mel stats: ",(mean,std))
        # print("for mel: ", (mean.shape,std.shape))
        # stats_dict['mel']['mean']=str(mean)
        # stats_dict['mel']['std']=str(std)
        np.savez_compressed('arabic_mel_stats', a=mean, b=std)