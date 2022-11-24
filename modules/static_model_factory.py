import functools
from pyexpat import features
import tensorflow as tf
from tensorflow.python.ops.signal import window_ops
from modules.tf_layers import ConvNormRelu, UpSampling1D
import numpy as np
import logging
logging.basicConfig(filename="log/debug_audio2keypoint1107.log", level=logging.INFO,filemode="w")
logging.info("getting logging ready")
# tf.enable_eager_execution()
# def tf_mel_spectograms(x_audio):
#     stft = tf.contrib.signal.stft(
#         x_audio,
#         400,
#         160, # originally 160, reduces the upsampling discrepancy in encoder
#         fft_length=512,
#         window_fn=functools.partial(window_ops.hann_window, periodic=True),
#         pad_end=False,
#         name=None
#     )
#     stft = tf.abs(stft)
#     mel_spect_input = tf.contrib.signal.linear_to_mel_weight_matrix(
#         num_mel_bins=64,
#         num_spectrogram_bins=tf.shape(stft)[2],
#         sample_rate=16000,
#         lower_edge_hertz=125.0,
#         upper_edge_hertz=7500.0,
#         dtype=tf.float32,
#         name=None
#     )

#     input_data = tf.tensordot(stft, mel_spect_input, 1)
#     input_data = tf.log(input_data + 1e-6)
#     #input_data = tf.signal.mfccs_from_log_mel_spectrograms(input_data)[:,:,:28]
#     input_data = tf.expand_dims(input_data, -1)

#     return input_data


def D_patchgan(x_pose, n_downsampling=2, norm='batch', reuse=False, is_training=False, scope='discriminator', debug=False):
    with tf.variable_scope(scope, reuse=reuse):
        ndf = 64
        model = tf.layers.conv1d(x_pose, filters=ndf, kernel_size=4, strides=2, padding='same',
                                 kernel_initializer=tf.glorot_uniform_initializer(),
                                 bias_initializer=tf.zeros_initializer(), activation=None)
        model = tf.nn.leaky_relu(model, alpha=0.2)

        for n in range(1, n_downsampling):
            nf_mult = min(2**n, 8)
            model = ConvNormRelu(model, ndf * nf_mult, type='1d', downsample=True, norm=norm,
                    leaky=True, is_training=is_training)

        nf_mult = min(2**n_downsampling, 8)
        model = ConvNormRelu(model, ndf * nf_mult, type='1d', k=4, s=1,
                norm=norm, leaky=True)

        model = tf.layers.conv1d(model, filters=1, kernel_size=4, strides=1,
                                 padding='same', kernel_initializer=tf.glorot_uniform_initializer(),
                                 bias_initializer=tf.zeros_initializer(), activation=None)
        if debug:
            print('discriminator model output size', model, 'scope', scope, 'reuse', reuse)
        return model

def G_audio2keypoints(input_dict, reuse=False, is_training=False, debug=False):
    with tf.variable_scope('generator', reuse=reuse):
        norm = input_dict['args'].norm
        x_audio = input_dict['audio']
        
        # features = input_dict['pitch']
        # fetaures = input_dict['gemaps']

        # mel_spec = tf_mel_spectograms(x_audio)
        # pitch_stats = np.load('/media/ssd/sulakshana/facesync_sulak/pitch_stats.npz')
        # mel_stats = np.load('/media/ssd/sulakshana/facesync_sulak/mel_stats.npz')
        # egmaps_stats = np.load('/media/ssd/sulakshana/facesync_sulak/egmaps_stats.npz')


        # pitch = tf.math.log(features+1e-6)
        # mel_mean = mel_stats['a'].astype(np.float32)
        # pitch_mean = pitch_stats['a'].astype(np.float32)
        # mel_std = mel_stats['b'].astype(np.float32)
        # pitch_std = pitch_stats['b'].astype(np.float32)

        # egmaps_mean = egmaps_stats['a'].astype(np.float32)
        # egmaps_std = egmaps_stats['b'].astype(np.float32)

        # normalise
        # mel_spec_norm = tf.math.divide(tf.math.subtract(mel_spec,tf.constant(mel_mean)),tf.constant(mel_std))
        # egmaps_spec_norm = tf.math.divide(tf.math.subtract(features,tf.constant(egmaps_mean)),tf.constant(egmaps_std))
        # pitch_norm = tf.math.divide(tf.math.subtract(pitch,tf.constant(pitch_mean)),tf.constant(pitch_std))
        
        # logging_strg = f"normalised min and max of mel: {(tf.math.reduce_min(mel_spec_norm),tf.math.reduce_max(mel_spec_norm))}"
        # logging.info(logging_strg)
        # logging_strg = f"normalised min and max of pitch: {(tf.math.reduce_min(pitch_norm),tf.math.reduce_max(pitch_norm))}"
        # print("normalised min and max of pitch: ", (tf.math.reduce_min(pitch_norm),tf.math.reduce_max(pitch_norm)))
        # logging.info(logging_strg)
        # n_mel = tf.shape(mel_spec.shape)[0].value
        # batch_size = 32
        # samp_size = 339
        # egmaps_spec_norm = tf.reshape(egmaps_spec_norm,shape = [1,int(-1),int(25),int(1)])
        # input_data = tf_mel_spectograms(x_audio)
        # input_data = tf.concat([mel_spec_norm,egmaps_spec_norm],axis = 2)
        
        # pitch_norm = tf.reshape(pitch_norm,shape = [1,int(-1),int(1),int(1)])
        # input_data = tf.concat([mel_spec_norm,pitch_norm],axis = 2)
        input_data = tf.expand_dims(x_audio, -1)
        # input_data = tf_mel_spectograms(x_audio)
        # input_data = egmaps_spec_norm 
        
        ref_landmarks = input_dict['ref_landmarks']
        piv_bottleneck = input_dict['piv_bottleneck']

        with tf.variable_scope('pose_variant_encoder', reuse=reuse):
            with tf.variable_scope('downsampling_block1'):
                conv = ConvNormRelu(ref_landmarks, 64, type='1d', leaky=True, downsample=False, norm=norm,
                                    is_training=is_training)
                first_block = ConvNormRelu(conv, 64, type='1d', leaky=True, downsample=True, norm=norm,
                                        is_training=is_training)

            with tf.variable_scope('downsampling_block2'):
                second_block = ConvNormRelu(first_block, 128, type='1d', leaky=True, downsample=False, norm=norm,
                                            is_training=is_training)
                second_block = ConvNormRelu(second_block, 128, type='1d', leaky=True, downsample=True, norm=norm,
                                            is_training=is_training)

            with tf.variable_scope('downsampling_block3'):
                third_block = ConvNormRelu(second_block, 256, type='1d', leaky=True, downsample=False, norm=norm,
                                        is_training=is_training)
                third_block = ConvNormRelu(third_block, 256, type='1d', leaky=True, downsample=True, norm=norm,
                                        is_training=is_training)

            with tf.variable_scope('downsampling_block4'):
                pv_bottleneck = ConvNormRelu(third_block, 256, type='1d', leaky=True, downsample=False, norm=norm,
                                            is_training=is_training)

        with tf.variable_scope('audio_encoder'):
            with tf.variable_scope('downsampling_block1'):
                conv = ConvNormRelu(input_data, 64, type='2d', leaky=True, downsample=False, norm=norm,
                                    is_training=is_training)
                first_block = ConvNormRelu(conv, 64, type='2d', leaky=True, downsample=True, norm=norm,
                                           is_training=is_training)

            with tf.variable_scope('downsampling_block2'):
                second_block = ConvNormRelu(first_block, 128, type='2d', leaky=True, downsample=False, norm=norm,
                                            is_training=is_training)
                second_block = ConvNormRelu(second_block, 128, type='2d', leaky=True, downsample=True, norm=norm,
                                            is_training=is_training)

            with tf.variable_scope('downsampling_block3'):
                third_block = ConvNormRelu(second_block, 256, type='2d', leaky=True, downsample=False, norm=norm,
                                           is_training=is_training)
                third_block = ConvNormRelu(third_block, 256, type='2d', leaky=True, downsample=True, norm=norm,
                                           is_training=is_training) 

            with tf.variable_scope('downsampling_block4'):
                fourth_block = ConvNormRelu(third_block, 256, type='2d', leaky=True, downsample=False, norm=norm,
                                            is_training=is_training)
                fourth_block = ConvNormRelu(fourth_block, 256, type='2d', leaky=True, downsample=False, norm=norm,
                                            is_training=is_training, k=(3, 8), s=1)#, padding='valid')

                fourth_block = tf.image.resize_bilinear(
                    fourth_block,
                    (input_dict["pose"].get_shape()[1].value, 1),
                    align_corners=False,
                    name=None
                )
                fifth_block = tf.squeeze(fourth_block, axis=2)

            with tf.variable_scope('downsampling_block5'):
                fifth_block = ConvNormRelu(fifth_block, 256, type='1d', leaky=True, downsample=False, norm=norm,
                                           is_training=is_training)
                fifth_block = ConvNormRelu(fifth_block, 256, type='1d', leaky=True, downsample=False, norm=norm,
                                           is_training=is_training)

                sixth_block = ConvNormRelu(fifth_block, 256, type='1d', leaky=True, downsample=True, norm=norm,
                                           is_training=is_training)

                seventh_block = ConvNormRelu(sixth_block, 256, type='1d', leaky=True, downsample=True, norm=norm,
                                             is_training=is_training)

                eight_block = ConvNormRelu(seventh_block, 256, type='1d', leaky=True, downsample=True, norm=norm,
                                           is_training=is_training)

                ninth_block = ConvNormRelu(eight_block, 256, type='1d', leaky=True, downsample=True, norm=norm,
                                           is_training=is_training)

                tenth_block = ConvNormRelu(ninth_block, 256, type='1d', leaky=True, downsample=True, norm=norm,
                                           is_training=is_training) # (1, 2, 256) for seq_len=64
                
                # concatenate pv and piv bottleneck features -> (1, 4, 256) for seq_len=64
                print("tenth_block shape:",tenth_block.shape)
                print("tenth_block shape:",pv_bottleneck.shape)
                print("tenth_block shape:",piv_bottleneck.shape)
                tenth_block = tf.concat([tenth_block, pv_bottleneck, piv_bottleneck], axis=1)
                tenth_block = tf.image.resize_bilinear(tenth_block[None, :, :, :], (ninth_block.shape[1].value, 1), align_corners=False, name=None)
                tenth_block = tf.squeeze(tenth_block, axis=2)

                ninth_block = tenth_block + ninth_block
                ninth_block = ConvNormRelu(ninth_block, 256, type='1d', leaky=True, downsample=False, norm=norm,
                                           is_training=is_training)

                eight_block = UpSampling1D(ninth_block) + eight_block
                eight_block = ConvNormRelu(eight_block, 256, type='1d', leaky=True, downsample=False, norm=norm,
                                           is_training=is_training)

                seventh_block = UpSampling1D(eight_block) + seventh_block
                seventh_block = ConvNormRelu(seventh_block, 256, type='1d', leaky=True, downsample=False, norm=norm,
                                             is_training=is_training)

                sixth_block = UpSampling1D(seventh_block) + sixth_block
                sixth_block = ConvNormRelu(sixth_block, 256, type='1d', leaky=True, downsample=False, norm=norm,
                                           is_training=is_training)

                fifth_block = UpSampling1D(sixth_block) + fifth_block
                audio_encoding = ConvNormRelu(fifth_block, 256, type='1d', leaky=True, downsample=False, norm=norm,
                                           is_training=is_training)


        with tf.variable_scope('decoder'):
            model = ConvNormRelu(audio_encoding, 256, type='1d', leaky=True, downsample=False, norm=norm, is_training=is_training)
            model = ConvNormRelu(model, 256, type='1d', leaky=True, downsample=False, norm=norm, is_training=is_training)
            model = ConvNormRelu(model, 256, type='1d', leaky=True, downsample=False, norm=norm, is_training=is_training)
            model = ConvNormRelu(model, 256, type='1d', leaky=True, downsample=False, norm=norm, is_training=is_training)

        with tf.variable_scope('logits'):
            model = tf.layers.conv1d(model, filters=136, kernel_size=1, strides=1,
                    kernel_initializer=tf.glorot_uniform_initializer(), bias_initializer=tf.zeros_initializer(),
                    padding='same', activation=None
                    )
    if debug:
        print('generator output size', model, 'reuse', reuse)
    return model

def E_piv(input_dict, reuse=False, is_training=False, debug=False):
    # TODO: try reshaping ref_landmarks to 2D and change the conv types below
    with tf.variable_scope('pose_invariant_encoder', reuse=reuse):
        ref_landmarks = input_dict['ref_landmarks']
        og_shape = ref_landmarks.get_shape().as_list()
        # reshape and back in order to get encoding for each frame if input is a sequence
        if int(og_shape[1]) > 1:
            ref_landmarks = tf.reshape(ref_landmarks, (-1, 1, int(og_shape[-1])))
        norm = input_dict['args'].norm
        with tf.variable_scope('downsampling_block1'):
            conv = ConvNormRelu(ref_landmarks, 64, type='1d', leaky=True, downsample=False, norm=norm,
                                is_training=is_training)
            first_block = ConvNormRelu(conv, 64, type='1d', leaky=True, downsample=True, norm=norm,
                                       is_training=is_training)

        with tf.variable_scope('downsampling_block2'):
            second_block = ConvNormRelu(first_block, 128, type='1d', leaky=True, downsample=False, norm=norm,
                                        is_training=is_training)
            second_block = ConvNormRelu(second_block, 128, type='1d', leaky=True, downsample=True, norm=norm,
                                        is_training=is_training)

        with tf.variable_scope('downsampling_block3'):
            third_block = ConvNormRelu(second_block, 256, type='1d', leaky=True, downsample=False, norm=norm,
                                       is_training=is_training)
            third_block = ConvNormRelu(third_block, 256, type='1d', leaky=True, downsample=True, norm=norm,
                                       is_training=is_training)

        with tf.variable_scope('downsampling_block4'):
            fourth_block = ConvNormRelu(third_block, 256, type='1d', leaky=True, downsample=False, norm=norm,
                                        is_training=is_training)
            piv_bottleneck = tf.reshape(fourth_block, tf.TensorShape((og_shape[0], og_shape[1], fourth_block.shape[-1])))
        if debug:
            print('pose invariant encoder output size', piv_bottleneck, 'reuse', reuse)
    return piv_bottleneck
