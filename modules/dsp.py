import os
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic
import opensmile
import parselmouth
from tensorflow.python.ops.signal import window_ops
import os
import tensorflow as tf
import functools
from preprocessing import normalise_acoustic_features
from python_speech_features import mfcc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_audio(file_path, sr=8000):
    '''
    load an audio file to mono and the specified sampling rate
    '''
    try:
        audio, fs = sf.read(file_path, dtype='float32')
    except Exception:
        _logger.info("Failed to extract audio with soundfile.")
        audio, fs = librosa.load(file_path)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    if fs != sr:
        audio = librosa.resample(audio, fs, target_sr=sr)
    return audio

def write_audio(audio, file_path, sr=8000):
    sf.write(file_path, audio, sr, subtype='PCM_16')

def walk_filter(input_dir, file_extension=None):
    files = []
    for r, _, fs in os.walk(input_dir, followlinks=True):
        if file_extension:
            files.extend([os.path.join(r, f) for f in fs if os.path.splitext(f)[-1] == file_extension])
        else:
            files.extend([os.path.join(r, f) for f in fs])
    return files
            
def filter_butter(data, lower_bound=None):
    T = 5.0
    fs = 10.0
    cutoff = 2
    nyq = 0.5 * fs
    order = 2
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    y = np.asarray(y)
    if lower_bound:
        y[y < lower_bound] = 0
    return y


def extract_mel_spec(signal):

    stft = tf.signal.stft(
        signal,
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

def extract_mel(signal):
    print("signal:",signal.shape)
    # signal = signal[None,:]
    stft = tf.signal.stft(
        signal,
        400, # frame_length , the window length
        160, # originally 160, reduces the upsampling discrepancy in encoder # frame_step , hop size
        fft_length= 512,
        window_fn= functools.partial(window_ops.hann_window, periodic=True),
        pad_end=False,
        name=None
    )
    print("stft",tf.shape(stft))
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


def extract_egemaps(signal):
    
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
    mel_spec = extract_mel_spec(signal)
    # num_pad = n-m
    n = signal.shape[0] 
    m = features.shape[0]
    num_pad = n-m
    # print("features.shape[0]: ",features.shape[0])
    # n_pad = 254 - features.shape[0]
    mel_norm_stats = np.load("/media/ssd/sulakshana/facesync_sulak/data/english_dataset/mel_spec_stats.npz")
    egemap_stats = np.load("/media/ssd/sulakshana/facesync_sulak/data/english_dataset/egemaps_stats.npz")
    
    features_val = np.concatenate((features,np.zeros((num_pad,features.shape[1])))) # zero pad pitch values to match dimensions
    features_val = normalise_acoustic_features(features_val,egemap_stats['mean'],egemap_stats['std'])
    mel_spec = normalise_acoustic_features(mel_spec,mel_norm_stats['mean'],mel_norm_stats['std'])
    acoustic_feature = np.concatenate((mel_spec,features_val),axis = 1)
    return acoustic_feature

def extract_pitch(signal):

    '''
    extract the pitch of an audio file given the path
    the algorithm used for pitch extraction is PYin
    '''

    print("extracting pitch")

    sr = 16000
    snd = parselmouth.Sound(signal,sampling_frequency = sr)
    pitch_parl = snd.to_pitch_ac()
    pitch = pitch_parl.selected_array['frequency']


    
 
    mel_spec = extract_mel_spec(signal)
    n = pitch.shape[0] 
    m = mel_spec.shape[0]
    num_pad = n-m
    pitch_val = np.concatenate((pitch,np.zeros(num_pad))) # zero pad pitch values to match dimensions 
    acoustic_feature = np.concatenate((mel_spec,pitch_val),axis = 1)

    return acoustic_feature

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
def extract_deepspeech_from_file(signal):
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
        # for audio_path in data_path:
        #     audio_file = np.load(audio_path)
        #     audio_file = dict(audio_file)
            audio = signal
            if audio.ndim != 1:
              print('Audio has multiple channels, only first channel is considered')
              audio = audio[0,:]
            input_vector = audioToInputVector(audio.astype('int16'), sample_rate, n_input, n_context)
            network_output = sess.run(layer_6, feed_dict={input_tensor: input_vector[np.newaxis, ...],
                                                                  seq_length: [input_vector.shape[0]]})
            print("network_output:",network_output.shape)
    
    network_output = network_output.squeeze()[None,:,:]
    mel_spec = extract_mel_spec(signal)
    n = network_output.shape[0] 
    m = mel_spec.shape[0]
    network_output = network_output[:,:m,:]
    mel_norm_stats = np.load("/media/ssd/sulakshana/facesync_sulak/data/english_dataset/mel_spec_stats.npz")
    deepspeech_stats = np.load("/media/ssd/sulakshana/facesync_sulak/data/english_dataset/deepspeech_stats.npz")

    features_val =  (network_output-deepspeech_stats['mean'])/deepspeech_stats['std']
    mel_spec = (mel_spec-mel_norm_stats['mean'])/mel_norm_stats['std']
    acoustic_feature = np.concatenate((mel_spec,features_val),axis = 1)
    return acoustic_feature

# def extract_pitch(signal):

#     '''
#     extract the pitch of an audio file given the path
#     the algorithm used for pitch extraction is PYin
#     '''

#     print("extracting pitch")
#     # sample = np.load(filepath)
#     # signal = sample['audio']

#     sr = 16000
#     snd = parselmouth.Sound(signal,sampling_frequency = sr)
#     pitch_parl = snd.to_pitch_ac()
#     pitch_val = pitch_parl.selected_array['frequency']
    
#     # pitch_val = np.concatenate((pitch,np.zeros(1))) # zero pad pitch values to match dimensions  
    
#     # print('out_file_path: ',out_file_path)
#     return pitch_val.reshape(1,-1)
