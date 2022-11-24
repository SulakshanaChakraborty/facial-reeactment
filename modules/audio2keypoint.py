import datetime
import logging
import subprocess
import logging

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from modules.config import get_config
from modules.dataset import DataGenerator
from modules.static_model_factory import D_patchgan, G_audio2keypoints, E_piv
from modules.tf_layers import to_motion_delta, keypoints_to_train, keypoints_regloss
from common.consts import POSE_SAMPLE_SHAPE, G_SCOPE, D_SCOPE, PIV_SCOPE, SR
from common.evaluation import compute_pck
from common.pose_logic_lib import translate_keypoints, get_sample_output_by_config, decode_pose_normalized_keypoints
from common.pose_plot_lib import save_side_by_side_video, save_video_from_audio_video, save_all_epochs_plot
from utils.dsp import filter_butter,extract_mel_spec
import utils.dsp
import tensorflow.contrib.slim as slim

logger = logging.getLogger('audio2keypoint')
logger.setLevel(logging.DEBUG)
_log_handler = logging.StreamHandler()
_log_handler.setLevel(logging.INFO)
_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_log_handler.setFormatter(_formatter)
logger.addHandler(_log_handler)

# TODO: remove when upgraded to TF2
tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)
import logging
logging.basicConfig(filename="log/debug_audio2keypoint.log", level=logging.INFO,filemode="w")
logging.info("getting logging ready")

class Audio2Keypoint():
    def __init__(self, args, seq_len=64, debug=False):
        self.args = args
        tf.reset_default_graph()
        if args.skip_normalisation:
            logger.info(f"Skipping data normalisation")
            self.data_mean = 0.0
            self.data_std = 1.0
        else:
            try:
                self.data_mean = np.load(f"{args.preprocessed_dir}/lm_mean.npy")
                self.data_std = np.load(f"{args.preprocessed_dir}/lm_std.npy")
            except:
                logger.warn(f"No normalisation files found in {args.preprocessed_dir}")
                self.data_mean = 0.0
                self.data_std = 1.0
        logging_strg= f" feature type mode: {args.feature_type}"
        logging.info(logging_strg)
        self.acoustic_stats = {}
        # if args.feature_type != 'deepspeech':
            
        if args.mel_spec_concat:
            logging_strg= f" concat mel? : {args.mel_spec_concat}"
            logging.info(logging_strg)
            mel_stats = dict(np.load(f"{args.preprocessed_dir}/mel_spec_stats.npz"))
            self.acoustic_stats['mel_spec'] = mel_stats
                
        file_name = args.feature_type +'_stats.npz'
        path = os.path.join(args.preprocessed_dir,file_name)
        features_stats = dict(np.load(path))
        self.acoustic_stats[args.feature_type] = features_stats
        
        # do not take all GPU memory if GPU available (doesn't seem to work correctly but 
        # export TF_FORCE_GPU_ALLOW_GROWTH=true does)
        gpu_config = None
        if tf.test.is_gpu_available():
            gpu_config = tf.ConfigProto()
            gpu_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=gpu_config)
        self.real_pose = tf.placeholder(tf.float32, [self.args.batch_size, seq_len, POSE_SAMPLE_SHAPE[-1]])
        self.ref_landmarks = tf.placeholder(tf.float32, [self.args.batch_size, 1, POSE_SAMPLE_SHAPE[-1]], name="ref_landmarks")
        if self.args.feature_type == 'mel_spec': feature_size = 64
        if self.args.feature_type == 'pitch': feature_size = 65
        if self.args.feature_type == 'egemaps': feature_size = 25
        if self.args.feature_type == 'deepspeech': feature_size = 29
            
        self.audio_A = tf.placeholder(tf.float32, [self.args.batch_size,None, feature_size], name="input_audio")
        if self.args.mode == "inference":
            self.audio_A = tf.placeholder(tf.float32, [self.args.batch_size,None, feature_size], name="input_audio")

        # self.pitch_B = tf.placeholder(tf.float32, [self.args.batch_size, None], name="input_pitch")
        self.is_training = tf.placeholder(tf.bool, (), name="is_training")
        cfg = get_config(self.args.config)

        self.piv_bottleneck = E_piv(
            {
            "ref_landmarks": self.ref_landmarks,
            "args": self.args
            },
            is_training=self.is_training,
            debug=debug)

        self.fake_pose = G_audio2keypoints(
            {
                "audio": self.audio_A,
                "pose": self.real_pose,
                "config": cfg,
                "args": self.args,
                "ref_landmarks": self.ref_landmarks,
                "piv_bottleneck": self.piv_bottleneck
                # "pitch":self.pitch_B
            },
            is_training=self.is_training,
            debug=debug)
        
        # just need fake pose and piv_bottleneck for inference
        if self.args.mode != 'inference':
            tf.summary.histogram("fake_pose", self.fake_pose, collections=['g_summaries'])

            # remove base keypoint which is always [0,0]. Keeping it may ruin GANs training due discrete problems. etc. TODO: verify
            training_keypoints = self._get_training_keypoints()

            training_real_pose = keypoints_to_train(self.real_pose, training_keypoints)
            training_real_pose = get_sample_output_by_config(training_real_pose, cfg)

            training_fake_pose = keypoints_to_train(self.fake_pose, training_keypoints)

            # regression loss on pose
            self.reg_loss = 0
            if self.args.reg_loss in ['pose', 'both']:
                pose_reg = keypoints_regloss(training_real_pose, training_fake_pose, self.args.reg_loss_type)
                tf.summary.scalar(name='pose_reg', tensor=pose_reg, collections=['g_summaries'])
                self.reg_loss += pose_reg
            
            if self.args.reg_loss in ['motion', 'both']:
                training_real_pose_motion = to_motion_delta(training_real_pose)
                training_fake_pose_motion = to_motion_delta(training_fake_pose)
                motion_reg = keypoints_regloss(training_real_pose_motion, training_fake_pose_motion,
                                            self.args.reg_loss_type) * self.args.lambda_motion_reg_loss
                tf.summary.scalar(name='motion_reg', tensor=motion_reg, collections=['g_summaries'])
                self.reg_loss += motion_reg

            # get full body keypoints
            D_training_keypoints = self._get_training_keypoints()
            D_real_pose = keypoints_to_train(self.real_pose, D_training_keypoints)
            D_fake_pose = keypoints_to_train(self.fake_pose, D_training_keypoints)

            # d motion or pose
            if self.args.reg_loss == 'motion':
                D_fake_pose_input = to_motion_delta(D_fake_pose)
                D_real_pose_input = to_motion_delta(D_real_pose)
            elif self.args.reg_loss == 'pose':
                D_fake_pose_input = D_fake_pose
                D_real_pose_input = D_real_pose
            elif self.args.reg_loss == 'both':
                # concatenate on the temporal axis
                D_fake_pose_input = tf.concat([D_fake_pose, to_motion_delta(D_fake_pose)], axis=1)
                D_real_pose_input = tf.concat([D_real_pose, to_motion_delta(D_real_pose)], axis=1)
            else:
                raise ValueError("d_input wrong value")

            self.fake_pose_score = D_patchgan(D_fake_pose_input, is_training=self.is_training, debug=True)
            self.real_pose_score = D_patchgan(D_real_pose_input, reuse=True, is_training=self.is_training, debug=True)
            # infp_str = f"fake_pose_score val: {self.fake_pose_score}"
            #logging.info(infp_str)
            tf.summary.histogram("fake_pose_score", self.fake_pose_score, collections=['d_summaries'])
            tf.summary.histogram("real_pose_score", self.real_pose_score, collections=['d_summaries'])

            # loss for training the global D
            self.D_loss = tf.losses.mean_squared_error(tf.ones_like(self.real_pose_score), self.real_pose_score) \
                            + tf.losses.mean_squared_error(tf.zeros_like(self.fake_pose_score),
                                                                            self.fake_pose_score)
            tf.summary.scalar(name='D_loss', tensor=self.D_loss, collections=['d_summaries'])

            # loss for training the generator from the global D - have I fooled the global D?
            self.adv_loss = tf.losses.mean_squared_error(tf.ones_like(self.fake_pose_score), self.fake_pose_score)
            tf.summary.scalar(name='adv_loss', tensor=self.adv_loss, collections=['g_summaries'])

            # train global D
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=D_SCOPE)):
                self.train_D = tf.train.AdamOptimizer(learning_rate=self.args.lr_d)\
                    .minimize(loss=self.D_loss,var_list=tf.trainable_variables(scope=D_SCOPE))

            # piv loss
            self.piv_real_pose_bottleneck = E_piv({"ref_landmarks": D_real_pose, "args": self.args}, reuse=True, is_training=self.is_training) # e(y)
            self.piv_fake_pose_bottleneck = E_piv({"ref_landmarks": D_fake_pose, "args": self.args},
                                                    reuse=True,
                                                    is_training=self.is_training)  # e(y)
            tf.summary.histogram("piv_fake_pose_bottleneck", self.piv_fake_pose_bottleneck, collections=['piv_summaries'])
            tf.summary.histogram("piv_real_pose_bottleneck", self.piv_real_pose_bottleneck, collections=['piv_summaries'])
            real_k = tf.repeat(self.ref_landmarks, repeats=int(self.piv_real_pose_bottleneck.shape[1]), axis=1)
            real_MSE = tf.losses.mean_squared_error(
                real_k,
                D_real_pose)
            real_L2 = tf.sqrt(real_MSE)
            tf.summary.histogram("real_k", real_k, collections=['piv_summaries'])
            tf.summary.scalar("real_MSE", tensor=real_MSE, collections=['piv_summaries'])
            tf.summary.scalar("real_L2", tensor=real_L2, collections=['piv_summaries'])
            fake_k = tf.repeat(self.ref_landmarks, repeats=int(self.piv_fake_pose_bottleneck.shape[1]), axis=1)
            fake_MSE = tf.losses.mean_squared_error(
                    fake_k,
                    D_fake_pose)
            fake_L2 = tf.sqrt(fake_MSE)
            tf.summary.histogram("fake_k", fake_k, collections=['piv_summaries'])
            tf.summary.scalar("fake_MSE", tensor=fake_MSE, collections=['piv_summaries'])
            tf.summary.scalar("fake_L2", tensor=fake_L2, collections=['piv_summaries'])
            self.piv_piv_loss = real_L2 + fake_L2 + tf.keras.backend.epsilon()  # L_{PIV-PIV}
            tf.summary.scalar(name='piv_piv_loss', tensor=self.piv_piv_loss, collections=['piv_summaries'])
            self.piv_loss = self.reg_loss + self.adv_loss + self.piv_piv_loss
            tf.summary.scalar(name='piv_loss', tensor=self.piv_loss, collections=['piv_summaries'])

            # train piv
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=PIV_SCOPE)):
                self.train_PIV = tf.train.AdamOptimizer(learning_rate=self.args.lr_piv)\
                    .minimize(loss=self.piv_loss,var_list=tf.trainable_variables(scope=PIV_SCOPE))

            fake_avg = tf.expand_dims(tf.reduce_mean(self.piv_fake_pose_bottleneck, axis=1), axis=1)
            tf.summary.histogram("piv_bottleneck", self.piv_bottleneck, collections=['piv_summaries'])
            tf.summary.histogram("fake_avg", fake_avg, collections=['piv_summaries'])
            self.piv_gen_loss = tf.sqrt(tf.losses.mean_squared_error(self.piv_bottleneck, fake_avg))  # L_{PIV-GEN}
            tf.summary.scalar(name='piv_gen_loss', tensor=self.piv_gen_loss, collections=['g_summaries'])

            # sum up ALL the losses for training the generator
            self.G_loss = self.reg_loss + self.adv_loss + self.piv_gen_loss
            tf.summary.scalar(name='train_loss', tensor=self.G_loss, collections=['g_summaries'])

            # train the generator
            trainable_variables = tf.trainable_variables(scope=G_SCOPE)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=G_SCOPE)):
                self.train_G = tf.train.AdamOptimizer(learning_rate=self.args.lr_g).minimize(
                    loss=self.G_loss, var_list=trainable_variables)
        
        if hasattr(self.args, 'checkpoint') and self.args.checkpoint:
            scope_list = ['generator', 'discriminator', 'pose_invariant_encoder']
            print("loading checkpoint..")
            self.restore(self.args.checkpoint, scope_list=scope_list)

    def _get_training_keypoints(self):
        return np.arange(0, 68, 1)  # 68 facial keypoints

    def inference(self, test_files):

        test_generator = DataGenerator(test_files, self.data_mean, self.data_std, self.acoustic_stats,batch_size = self.args.batch_size, shuffle=False,mode = self.args.feature_type,concat_mel = self.args.mel_spec_concat)
        # test_batch = next(test_generator)
        print("test files batch: ",len(test_generator))
        keypoints1_list = []
        keypoints2_list = []
        avg_loss = []
        for i in range(len(test_generator)):
            # print("hello?????")
            audio_X, pose_Y = test_generator.__getitem__(i) 
            res, loss = self.sess.run([self.fake_pose, self.G_loss],
                                            feed_dict={
                                                self.audio_A: audio_X,
                                                self.is_training: 0,
                                                self.real_pose: pose_Y,
                                                self.ref_landmarks: pose_Y[:, 0, :][:, None, :] # first landmark of each sequence
                                                #   self.pitch_B: pitch
                                            })

            keypoints1 = decode_pose_normalized_keypoints(res, self.data_mean, self.data_std, [0, 0])
            keypoints2 = decode_pose_normalized_keypoints(pose_Y, self.data_mean, self.data_std, [0, 0])
            keypoints1_list.append(keypoints1)
            keypoints2_list.append(keypoints2)
            avg_loss.append(loss)

        keypoints1_list = np.array(keypoints1_list).reshape((-1, 2, 68))
            
        #logging_strg= f" output of keypoints1_list arr shape: {keypoints1_list.shape}"
        #logging.info(logging_strg)
        
        keypoints2_list = np.array(keypoints2_list).reshape((-1, 2, 68))

        pred = np.delete(keypoints1_list, 33, 2)  # don't compare base point
        gt = np.delete(keypoints2_list, 33, 2)
        pcks = compute_pck(pred, gt)
        pck_loss = np.mean(pcks)
        abs_error = np.absolute(gt - pred)
        l1_loss = np.mean(abs_error.flatten())
        test_loss = np.mean(avg_loss)
        print("--------------------------- test statistics ---------------------------")
        print(
            f"loss: {test_loss:.4f}, l1_loss: {l1_loss:.4f}, pck: {pck_loss:.4f}"
        )
        print("-----------------------------------------------------------------------")
       
    def train(self, train_files, val_files, batch_size=32, epochs=1000):
        train_generator = DataGenerator(train_files, self.data_mean, self.data_std, self.acoustic_stats,batch_size = batch_size,mode = self.args.feature_type,concat_mel = self.args.mel_spec_concat)
        val_generator = DataGenerator(val_files, self.data_mean, self.data_std, self.acoustic_stats,batch_size = batch_size, shuffle=False,mode = self.args.feature_type,concat_mel = self.args.mel_spec_concat)
        #logging_strg = f" len of val files: {len(val_files)}"
        #logging.info(logging_strg)

        #logging_strg = f" val files: {val_files}"
        #logging.info(logging_strg)
        # prep dirs
        checkpoint_dir = f"{self.args.output_path}/checkpoints"
        tf_logs_dir = f"{self.args.output_path}/tf_logs"
        val_samples_dir = f"{self.args.output_path}/val_samples"
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(tf_logs_dir, exist_ok=True)
        os.makedirs(val_samples_dir, exist_ok=True)

        cfg = get_config(self.args.config)

        self.sess.run(tf.global_variables_initializer())
        if hasattr(self.args, 'checkpoint') and self.args.checkpoint:
            scope_list = ['generator', 'discriminator', 'pose_invariant_encoder']
            self.restore(self.args.checkpoint, scope_list=scope_list)

        g_summaries = tf.summary.merge(tf.get_collection("g_summaries"))
        d_summaries = tf.summary.merge(tf.get_collection("d_summaries"))
        piv_summaries = tf.summary.merge(tf.get_collection("piv_summaries"))

        writer = tf.summary.FileWriter(tf_logs_dir, self.sess.graph)
        ITR_PER_EPOCH = np.min([1000, len(train_generator)]) # TODO: experiment with this
        hit_nan = False

        # init
        lowest_validation_loss = 10
        for j in range(epochs):

            if hit_nan:
                break

            keypoints1_list = []
            keypoints2_list = []
            val_loss = []
            #logging_strg= f" len of val generator: {len(val_generator)}"
            #logging.info(logging_strg)
            for i in range(len(val_generator)):
                audio_X, pose_Y = val_generator.__getitem__(i) # removed pitch
                res, loss = self.sess.run([self.fake_pose, self.G_loss],
                                          feed_dict={
                                              self.audio_A: audio_X,
                                              self.is_training: 0,
                                              self.real_pose: pose_Y,
                                              self.ref_landmarks: pose_Y[:, 0, :][:, None, :] # first landmark of each sequence
                                            #   self.pitch_B: pitch
                                          })
                #logging_strg= f" output of val result shape: {res.shape}"
                #logging.info(logging_strg)

                #logging_strg= f" output of val result: {res}"
                #logging.info(logging_strg)

                #logging_strg= f" output of val loss: {loss}"
                #logging.info(logging_strg)
                # returns reshaped keypoints (2048, 2, 68)
                keypoints1 = decode_pose_normalized_keypoints(res, self.data_mean, self.data_std, [0, 0])
                keypoints2 = decode_pose_normalized_keypoints(pose_Y, self.data_mean, self.data_std, [0, 0])
                keypoints1_list.append(keypoints1)
                keypoints2_list.append(keypoints2)
                val_loss.append(loss)
            avg_loss = np.mean(val_loss)

            #logging_strg= f" output of keypoints1_list list shape: {len(keypoints1_list)}"
            #logging.info(logging_strg)
            
            keypoints1_list = np.array(keypoints1_list).reshape((-1, 2, 68))
            
            #logging_strg= f" output of keypoints1_list arr shape: {keypoints1_list.shape}"
            #logging.info(logging_strg)
            
            keypoints2_list = np.array(keypoints2_list).reshape((-1, 2, 68))

            pred = np.delete(keypoints1_list, 33, 2)  # don't compare base point
            gt = np.delete(keypoints2_list, 33, 2)
            pcks = compute_pck(pred, gt)
            pck_loss = np.mean(pcks)
            abs_error = np.absolute(gt - pred)
            l1_loss = np.mean(abs_error.flatten())
            print(
                f"Epoch {j} (step {ITR_PER_EPOCH * j}): val_loss: {avg_loss:.4f}, l1_loss: {l1_loss:.4f}, pck: {pck_loss:.4f}"
            )
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="val_loss", simple_value=avg_loss),
            ])
            writer.add_summary(summary, global_step=ITR_PER_EPOCH * j)
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="l1_loss", simple_value=l1_loss),
            ])
            writer.add_summary(summary, global_step=ITR_PER_EPOCH * j)
            summary = tf.Summary(value=[
                tf.Summary.Value(tag="pck_loss", simple_value=pck_loss),
            ])
            writer.add_summary(summary, global_step=ITR_PER_EPOCH * j)

            if lowest_validation_loss > avg_loss:
                lowest_validation_loss = avg_loss
                tf.train.Saver().save(
                    self.sess, f'{checkpoint_dir}/best_ckpt-step_{ITR_PER_EPOCH * j}-loss_{avg_loss:3f}l1_{l1_loss:.3f}-pck_{pck_loss:.3f}.ckp')

            if j % 15 == 14:
                tf.train.Saver().save(self.sess, f'{checkpoint_dir}/ckpt-step-{ITR_PER_EPOCH * j}.ckp')
                if self.args.output_videos:
                    keypoints1_list = keypoints1_list.reshape((len(val_generator), 32, 64, 2, 68))  # tmp
                    keypoints2_list = keypoints2_list.reshape((len(val_generator), 32, 64, 2, 68))
                    #logging_strg= f" output of keypoints1_list shape: {keypoints1_list.shape}"
                    #logging.info(logging_strg)
                    pcks = pcks.reshape((len(val_generator), 32, 64))
                    abs_error = abs_error.reshape(len(val_generator), 32, 64, 2, 67)
                    self.save_prediction_video(val_files,
                                               keypoints1_list,
                                               keypoints2_list,
                                               f"{val_samples_dir}/{j}",
                                               limit=5,
                                               loss=avg_loss,
                                               pcks=pcks,
                                               abs_error=abs_error)
            minibatch_g_loss = []
            minibatch_loss = []
            for i in range(ITR_PER_EPOCH):
                audio_X, pose_Y = train_generator.__getitem__(i) # removed pitch
                # audio_X, pose_Y = train_generator.__getitem__(i)
                # D
                #logging.info("getting logging ready")
                # logging_str = f"len of train gen {len(train_generator)}"
                #logging.info(logging_str)

                
                # logging_str = f"audio_X shape: {audio_X.shape}"
                #logging.info(logging_str)

                # logging_str = f"pose_Y shape: {pose_Y.shape}"
                #logging.info(logging_str)

                # logging_str = f"pitch shape: {pitch.shape}"
                #logging.info(logging_str)

                d_loss, d_summaries_str, _ = self.sess.run([self.D_loss, d_summaries, self.train_D],
                                                           feed_dict={
                                                               self.audio_A: audio_X,
                                                               self.real_pose: pose_Y,
                                                               self.is_training: 1,
                                                               self.ref_landmarks: pose_Y[:, 0, :][:, None, :]
                                                            #    self.pitch_B: pitch
                                                           })
                logging_str = f"d_loss loss: {d_loss}"
                #logging.info(logging_str)

                writer.add_summary(d_summaries_str, global_step=(ITR_PER_EPOCH * j) + i)

                # G
                g_loss, g_regloss, g_adv_loss, piv_gen_loss, g_summaries_str, _, piv_piv_loss, piv_loss, piv_summaries_str, _ = self.sess.run(
                    [self.G_loss, self.reg_loss, self.adv_loss, self.piv_gen_loss, g_summaries, self.train_G, self.piv_piv_loss, self.piv_loss, piv_summaries, self.train_PIV],
                    feed_dict={
                        self.audio_A: audio_X,
                        self.real_pose: pose_Y,
                        self.is_training: 1,
                        self.ref_landmarks: pose_Y[:, 0, :][:, None, :]
                        # self.pitch_B: pitch
                    })
                logging_str = f"g_loss loss: {g_loss}"
                #logging.info(logging_str)

                minibatch_g_loss.append(g_loss)
                minibatch_loss.append(g_regloss)

                if np.isnan(g_loss) or np.isnan(g_regloss) or np.isnan(g_adv_loss) or np.isnan(piv_gen_loss) or np.isnan(piv_piv_loss) or np.isnan(piv_loss):
                    writer.add_summary(g_summaries_str, global_step=(ITR_PER_EPOCH * j) + i)
                    writer.add_summary(piv_summaries_str, global_step=(ITR_PER_EPOCH * j) + i)
                    print(
                        f"Epoch {j}: Iteration: {i}. g_loss: {np.mean(minibatch_g_loss):.4f}, g_adv_loss: {g_adv_loss:.4f}, g_reg_loss: {np.mean(minibatch_loss):.4f}"
                    )
                    print(f"Epoch {j}: Iteration: {i}, piv_gen_loss: {piv_gen_loss:.4f}, piv_piv_loss: {piv_piv_loss:.4f}, piv_loss: {piv_loss:.4f}") # TODO: figure out if minibatch makes more sense
                    print(f"Epoch {j}: Iteration: {i}. d_loss: {d_loss:.4f}")
                    print("Ending training since we hit nans")
                    hit_nan = True
                    break


                if i % 100 == 0:
                    writer.add_summary(g_summaries_str, global_step=(ITR_PER_EPOCH * j) + i)
                    writer.add_summary(piv_summaries_str, global_step=(ITR_PER_EPOCH * j) + i)
                    print(
                        f"Epoch {j}: Iteration: {i}. g_loss: {np.mean(minibatch_g_loss):.4f}, g_adv_loss: {g_adv_loss:.4f}, g_reg_loss: {np.mean(minibatch_loss):.4f}"
                    )
                    print(f"Epoch {j}: Iteration: {i}, piv_gen_loss: {piv_gen_loss:.4f}, piv_piv_loss: {piv_piv_loss:.4f}, piv_loss: {piv_loss:.4f}") # TODO: figure out if minibatch makes more sense
                    print(f"Epoch {j}: Iteration: {i}. d_loss: {d_loss:.4f}")

            train_generator.on_epoch_end()
        
        save_all_epochs_plot(val_samples_dir, f"{val_samples_dir}/epochs_plot")

        # print total number of model parameters
        total_parameters = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
        print("total number of trainable parameters:",total_parameters)

        # print model summary
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)


    def restore(self, ckp, scope_list=('generator')):
        variables = []
        for s in scope_list:
            variables += tf.global_variables(scope=s)
        tf.train.Saver(variables).restore(self.sess, ckp)

    def save_prediction_video(self,
                              val_files,
                              keypoints_pred,
                              keypoints_gt,
                              save_path,
                              limit=None,
                              loss=None,
                              pcks=None,
                              abs_error=None):
        n_batches = keypoints_pred.shape[1]
        for i in range(min(len(val_files), limit)):
            try:
                file_name = os.path.basename(val_files[i]).split(".npz")[0]
                
                #logging_strg= f"1) keypoint pred shape: {keypoints_pred.shape}"
                #logging.info(logging_strg)
                
                #logging_strg= f"2) keypoints pred  {keypoints_pred}"
                #logging.info(logging_strg)

                #logging_strg= f"3) keypoints gt  shape {keypoints_gt.shape}"
                #logging.info(logging_strg)
                
                keypoints1 = keypoints_pred[i // n_batches, i % n_batches, :, :, :]
                keypoints2 = keypoints_gt[i // n_batches, i % n_batches, :, :, :]
                pck = np.mean(pcks[i // n_batches, i % n_batches, :])
                l1_loss = np.mean(abs_error[i // n_batches, i % n_batches, :, :, :])
                
                # save arrays for later joint visualizations
                if not (os.path.exists(save_path)):
                    os.makedirs(save_path)
                audio = np.load(val_files[i])['audio']
                output_path = f"{save_path}/{file_name}_{loss:.2f}_{l1_loss:.2f}_{pck:.3f}"
                np.savez(output_path, keypoints1=keypoints1, keypoints2=keypoints2, audio=audio)
                logger.info(f"Saved validation synthesis to {output_path}")
            except Exception as e:
                logger.exception(e)

    def predict_audio(self, signal, ref_landmarks, padded_pose_shape=None, shift_pred=(0, 0)):
        pose_shape = int(self.args.fps * float(signal.shape[0]) / self.args.sr)
        if not padded_pose_shape:
            padded_pose_shape = pose_shape + (2**5) - pose_shape % (2**5)
        padded_audio_shape = int(padded_pose_shape * self.args.sr / self.args.fps)
        padded_audio = np.pad(signal, [0, padded_audio_shape - signal.shape[0]], mode='reflect')
        
        if self.args.feature_type == "mel_spec":
            feature = utils.dsp.extract_mel(padded_audio)
        if self.args.feature_type == "pitch":
            feature = utils.dsp.extract_pitch(padded_audio)
        if self.args.feature_type == "egemaps":
            feature = utils.dsp.extract_egemaps(padded_audio)
        if self.args.feature_type == "deepspeech":
            feature = utils.dsp.extract_deepspeech(padded_audio)

        print("features shape: ",feature.shape)

        res = self.sess.run(self.fake_pose, feed_dict={self.audio_A: feature[None,:,:], self.ref_landmarks: ref_landmarks.reshape(-1)[None, None, :], self.is_training: 0})
        keypoints = decode_pose_normalized_keypoints(res, self.data_mean, self.data_std, shift_pred)[:pose_shape]
        # shift nose
        keypoints_t = keypoints.transpose(0, 2, 1)
        keypoints_t[:, :, :] += ref_landmarks[33:34, :]
        # filter across time
        filtered_keypoints = keypoints_t.copy().transpose(1, 2, 0)
        for pi, pv in enumerate(filtered_keypoints):
            for ci, cv in enumerate(filtered_keypoints[pi]):
                filtered_keypoints[pi][ci] = filter_butter(filtered_keypoints[pi][ci], lower_bound=0.005)

        return filtered_keypoints.transpose(2, 0, 1)