import argparse
import os
import concurrent.futures as ft
import numpy as np
import parselmouth
import opensmile
import functools
from tensorflow.python.ops.signal import window_ops
import os
import tensorflow as tf
from python_speech_features import mfcc
from audio_handler import  AudioHandler
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def extract_pitch_from_file(signal):

    '''
    extract the pitch of an audio file given the path
    the algorithm used for pitch extraction is PYin
    '''

    print("extracting pitch")

    sr = 16000
    snd = parselmouth.Sound(signal,sampling_frequency = sr)
    pitch_parl = snd.to_pitch_ac()
    pitch = pitch_parl.selected_array['frequency']
    
    pitch_val = np.concatenate((pitch,np.zeros(1))) # zero pad pitch values to match dimensions  

    return pitch_val


def extract_egmaps_from_file(signal):

    '''
    extract eGMAPS fetures 
    '''

    print("extracting egmaps")

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    
    y = smile.process_signal(signal, 16000)
    features = y.to_numpy()
    n_pad = 254 - features.shape[0]
    features_val = np.concatenate((features,np.zeros((n_pad,25)))) # zero pad pitch values to match dimensions

    return features_val

def mel_spec_from_audio(res_audio):
    print("extracting mel")
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

    # input_data = input_data.eval(session=tf.compat.v1.Session())
    # res = input_data.numpy()
    return input_data

def extract_features_from_file(filepath,args):

    file = np.load(filepath)
    data = dict(file)

    if args.pitch == "y": 
        if 'pitch' in data.keys():
            return 
        pitch_val = np.log(extract_pitch_from_file(file['audio'])+1e-6)
        data['pitch'] = pitch_val
    if args.egemaps == "y": 
        if 'egemaps' in data.keys():
            return 
        egmaps_val = extract_egmaps_from_file(file['audio'])
        data['egemaps'] = egmaps_val

    # if args.mel_spec == "y": 
    #     mel_val_tf = mel_spec_from_audio(file['audio'])
    #     data['mel_spec'] = mel_val_tf
    # if args.deepspeech =="y":
    #     data['deepspeech'] = extract_deepspeech_from_file(file['audio'])
    np.savez_compressed(filepath,**data)
    
def walk_filter(input_dir, file_extension=None):
    files = []

    for r, _, fs in os.walk(input_dir, followlinks=True):
        if file_extension:
            files.extend([os.path.join(r, f) for f in fs if os.path.splitext(f)[-1] == file_extension])
        else:
            files.extend([os.path.join(r, f) for f in fs])

    return files

def audioToInputVector(audio, fs, numcep, numcontext):
        # Get mfcc coefficients
        features = mfcc(audio, samplerate=fs, numcep=numcep)

        # We only keep every second feature (BiRNN stride = 2)
        # features = features[::2]

        # One stride per time step in the input
        num_strides = len(features)

        # Add empty initial and final contexts
        empty_context = np.zeros((numcontext, numcep), dtype=features.dtype)
        features = np.concatenate((empty_context, features, empty_context))

        # Create a view into the array with overlapping strides of size
        # numcontext (past) + 1 (present) + numcontext (future)
        window_size = 2 * numcontext + 1
        train_inputs = np.lib.stride_tricks.as_strided(
            features,
            (num_strides, window_size, numcep),
            (features.strides[0], features.strides[0], features.strides[1]),
            writeable=False)
        # print("train_inputs:",train_inputs.shape)
        # Flatten the second and third dimensions
        train_inputs = np.reshape(train_inputs, [num_strides, -1])
        # print("train inputs reshape:",train_inputs.shape)

        train_inputs = np.copy(train_inputs)
        train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)

        # Return results
        return train_inputs

def extract_deepspeech_from_file(data_path):
    
    sample_rate = 16000 
    ds_path = "ds_graph/output_graph.pb"


    # tmp_audio = {'subj': {'seq': {'audio': audio, 'sample_rate': sample_rate}}}
    # audio_handler = AudioHandler(config)

    
        # Load graph and place_holders
    with tf.gfile.GFile(ds_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    graph = tf.get_default_graph()
    tf.import_graph_def(graph_def, name="deepspeech")
    input_tensor = graph.get_tensor_by_name('deepspeech/input_node:0')
    seq_length = graph.get_tensor_by_name('deepspeech/input_lengths:0')
    layer_6 = graph.get_tensor_by_name('deepspeech/logits:0')

    n_input = 26
    n_context = 9
    # processed_audio = copy.deepcopy(audio)
    with tf.Session(graph=graph) as sess:
        for audio_path in data_path:
            print("extracting deepspeech")
            audio_file = np.load(audio_path)
            audio_file = dict(audio_file)
            audio = audio_file['audio']
            if audio.ndim != 1:
              print('Audio has multiple channels, only first channel is considered')
              audio = audio[0,:]
            input_vector = audioToInputVector(audio.astype('int16'), sample_rate, n_input, n_context)
            network_output = sess.run(layer_6, feed_dict={input_tensor: input_vector[np.newaxis, ...],
                                                                  seq_length: [input_vector.shape[0]]})
            print("network_output:",network_output.shape)
            audio_file['deepspeech'] = network_output.squeeze()[None,:,:]
            np.savez_compressed(audio_path,**audio_file)

def extract_mel_from_file(file_names,args):
    audio = tf.placeholder(tf.float32,shape =[1,40960])
    tf_out = mel_spec_from_audio(audio)
    # running_sum = 0
    # running_sum_2 = 0
    with tf.Session() as sess:
        for file in file_names:
            
            # print("extract mel file id: ",idx)
            print(file)
            zip_file = np.load(file)
            data = dict(zip_file)
            print("extracting mel spec")
            
            if 'mel_spec' in data.keys():
                continue
            # aud = zip_file['audio']
            # print(aud.stype,aud.shape)
            
            mel_spec = sess.run(tf_out,feed_dict={audio:zip_file['audio']})
            
            data['mel_spec'] = mel_spec
            np.savez_compressed(file,**data)
            # running_sum += np.sum(mel_spec).squeeze()
            # running_sum_2 += np.sum(mel_spec**2).squeeze()
            # mel_spec_from_audio(zip_file['audio'])
            # mel_arr[idx] = mel_spec
    # _,m,l = mel_spec.shape
    # n = len(file_names)
    # _,samples,mel_bins = mel_spec.shape
    # num_elem = n*samples*mel_bins
    # mean_tot = running_sum/num_elem
    # std_tot = np.sqrt((running_sum_2 /num_elem) - (mean_tot **2)) 
    # file_save_path = os.path.join(args.input_dir ,'mel_spec_stats.npz')
    # print("Mel Spectogram stats shape",(mean_tot.shape,std_tot.shape))
    # np.savez_compressed(file_save_path, mean=mean_tot, std=std_tot)
    # print(mel_spec.shape)
    # print(mel_arr.shape)


def compute_egmaps_stats(audio_data_files,args):
    n = len(audio_data_files)
    egmaps_arr = np.zeros((n,254,25))
    for idx, data_path in enumerate (audio_data_files):
        file = np.load(data_path)
        arr = file['egemaps']
        egmaps_arr[idx] = arr
    egmaps_arr=egmaps_arr.reshape(-1,25)
    mean = np.mean(egmaps_arr,axis =0) 
    std = np.std(egmaps_arr,axis=0)
    file_save_path = os.path.join(args.input_dir ,'egemaps_spec_stats.npz')
    np.savez_compressed(file_save_path, mean=mean, std=std)

def compute_pitch_stats(audio_data_files,args):

    n = len(audio_data_files)
    pitch_arr = np.zeros((n,254))
    for idx, data_path in enumerate (audio_data_files):
        file = np.load(data_path)
        pitch_arr[idx] = file['pitch']
    #pitch_arr = np.log(pitch_arr +1e-6)
    mean = np.mean(pitch_arr) 
    std = np.std(pitch_arr)
    file_save_path = os.path.join(args.input_dir ,'pitch_spec_stats.npz')
    np.savez_compressed(file_save_path, mean=mean, std=std)


def compute_mel_stats(audio_data_files,args):
    n = len(audio_data_files)
    egmaps_arr = np.zeros((n,254,64))
    for idx, data_path in enumerate (audio_data_files):
        file = np.load(data_path)
        arr = file['mel_spec']
        egmaps_arr[idx] = arr
    egmaps_arr=egmaps_arr.reshape(-1,25)
    mean = np.mean(egmaps_arr,axis =0) 
    std = np.std(egmaps_arr,axis=0)
    file_save_path = os.path.join(args.input_dir ,'egemaps_spec_stats.npz')
    np.savez_compressed(file_save_path, mean=mean, std=std)

def compute_stats(audio_data_files,args):
    n = len(audio_data_files)
    mel_arr = np.zeros((n,254,64))
    # deepspeech_arr = np.zeros((n,254,29))
    egmaps_arr = np.zeros((n,254,25))
    pitch_arr = np.zeros((n,254))
    for idx, data_path in enumerate (audio_data_files):
        file = np.load(data_path)
        mel_arr[idx] = file['mel_spec']
        # arr = file['deepspeech']
        # deepspeech_arr[idx] = arr[:,:254,:]
        egmaps_arr[idx] = file['egemaps']
        pitch_arr[idx] = file['pitch']
    
    egmaps_arr=egmaps_arr.reshape(-1,25)
    file_save_path = os.path.join(args.out_dir_stats ,'egemaps_stats.npz')
    np.savez_compressed(file_save_path, mean=np.mean(egmaps_arr,axis =0) , std=np.std(egmaps_arr,axis=0))
    

    file_save_path = os.path.join(args.out_dir_stats ,'mel_spec_stats.npz')
    np.savez_compressed(file_save_path, mean=np.mean(mel_arr) , std=np.std(mel_arr))

    # file_save_path = os.path.join(args.out_dir_stats ,'deepspeech_stats.npz')
    # np.savez_compressed(file_save_path, mean=np.mean(deepspeech_arr) , std=np.std(deepspeech_arr))

    file_save_path = os.path.join(args.out_dir_stats ,'pitch_stats.npz')
    np.savez_compressed(file_save_path, mean=np.mean(pitch_arr) , std=np.std(pitch_arr))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    input_dir  = "/media/ssd/sulakshana/facesync_sulak/data/arabic_dataset/data"
    out_dir_stats = "/media/ssd/sulakshana/facesync_sulak/data/arabic_dataset/"

    parser.add_argument('--input_dir', type = str, default = input_dir, help='path to the data')
    parser.add_argument('--pitch', type = str, default = "n", help='extract pitch')
    parser.add_argument('--egemaps', type = str, default = "n", help='extract opensmile feature set (eGeMAPs LLD)')
    parser.add_argument('--mel_spec', type = str, default = "y", help='extract mel spectogram')
    parser.add_argument('--compute_stats', type = str, default = "y", help='compute stats of features')
    parser.add_argument('--deepspeech', type = str, default = "n", help='compute stats of features')
    # parser.add_argument('--output_dir', type =str, default = output_dir, help='path to output dir')
    parser.add_argument('--n_jobs', type=int, default=8, help='number of parallel processes')
    args = parser.parse_args()
    args.out_dir_stats = out_dir_stats


    data_files = walk_filter(args.input_dir, ".npz")
    print("number of data files: ",len(data_files))
    jobs = []
    # if args.egemaps == 'y' or args.pitch == 'y':
    # with ft.ProcessPoolExecutor(max_workers=args.n_jobs) as worker:
    #         for file_path in data_files:
    #             jobs.append(
    #                 worker.submit(extract_features_from_file,file_path,args)
    #             )
    #     if args.egemaps == 'y': compute_egmaps_stats(data_files)
    #     if args.pitch == 'y': compute_pitch_stats(data_files)

    # for file_path in data_files:
    # if args.pitch == "y" or args.egemaps =="y":
    #     # for file_path in data_files:
    #     #     extract_features_from_file(file_path,args)
    #     with ft.ProcessPoolExecutor(max_workers=args.n_jobs) as worker:
    #         for file_path in data_files:
    #             jobs.append(
    #                 worker.submit(extract_features_from_file,file_path,args)
    #             )
    

    # if args.mel_spec == 'y':
    #     extract_mel_from_file(data_files,args)

    # # compute tf graph
    # if args.deepspeech == 'y':
    #     extract_deepspeech_from_file(data_files)

    if args.compute_stats == 'y':
        train_path = os.path.join(args.input_dir,"train")
        train_files = walk_filter(train_path, ".npz")
        compute_stats(train_files,args)
    
    print("completed!")
