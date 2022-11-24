from common.pose_logic_lib import normalize_relative_keypoints, preprocess_to_relative
import numpy as np
import tensorflow as tf
import keras
import scipy.sparse
import os
import sys
import json
import threading
from preprocessing import normalise_acoustic_features
# from utils.dsp import extract_pitch

# import logging
# logging.basicConfig(filename="log/debug_dataset2507_NOW.log", level=logging.INFO,filemode="w")
# logging.info("getting logging ready dataset.py")

class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, mean, std,acoustic_ft_norm_stats, batch_size=32, shuffle=True,mode = 'baseline',concat_mel = True):
        self.list_IDs = list_IDs
        self.data_mean = mean
        self.data_std = std
        self.batch_size = batch_size
        self.norm_stats_acoustic= acoustic_ft_norm_stats
        # logging_str = f" batch size : {batch_size}"
        # logging.info(logging_str)
        self.shuffle = shuffle
        self.mode = mode
        self.concat_mel = concat_mel
        self.on_epoch_end()

    def __iter__(self):
        return self

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        #logging_str = f" list_IDs_temp : {list_IDs_temp}"
        #logging.info(logging_str)

        # Generate data
        visual_features = np.zeros((self.batch_size, 64, 68 * 2))
        if self.mode == 'pitch': acoustic_features =  np.zeros((self.batch_size, 254,65))
        if self.mode == 'mel_spec': acoustic_features =  np.zeros((self.batch_size, 254,64))
        if self.mode == 'egemaps':  acoustic_features = np.zeros((self.batch_size, 254,25))
        if self.mode == 'deepspeech':  acoustic_features = np.zeros((self.batch_size, 254,29))
        if self.concat_mel == True: mel_features = np.zeros((254,64))


        for idx, file_name in enumerate(list_IDs_temp):
            if file_name.lower().endswith('.npz'):
                sample = np.load(file_name)
            else:
                print("Unknown file type!")
                sys.exit(0)

            relative_keypoints = preprocess_to_relative(sample["pose"])
            visual_features[idx] = normalize_relative_keypoints(relative_keypoints, self.data_mean, self.data_std)
            val = sample[self.mode]
            if self.mode == 'deepspeech':
                ft = val[:,:254,:].squeeze()
              
            elif self.mode == 'mel_spec':
                acoustic_features[idx] = val #,self.norm_stats_acoustic[self.mode])

            elif self.mode == 'egemaps':
                # print(np.mean(val,axis =0).shape)
                # print(np.std(val,axis =0).shape)
                ft = normalise_acoustic_features(sample[self.mode] ,self.norm_stats_acoustic[self.mode])
                # print("egemaps",np.sum(ft))
                # print("egemaps shape",val.shape)
            elif self.mode == 'pitch':
                ft = normalise_acoustic_features(sample[self.mode] ,self.norm_stats_acoustic[self.mode])
            
            # ensure 2D input
            if len(val.squeeze().shape)<2:
                ft = val[:,np.newaxis]
             
            # if len(ft.squeeze().shape)>2:
            #     ft = ft.reshape(ft.shape[0],-1)
            
            if self.concat_mel: 
                # val = sample['mel_spec'].squeeze()
                # std = np.std(sample['mel_spec'].squeeze(),axis =1)
                # mean = np.mean(sample['mel_spec'].squeeze(),axis =1)
                # print(std.shape)
                # print(mean.shape)
                # print("mel spec shape",sample['mel_spec'].shape)
                # mel_features = (val - np.mean(val,axis =0))/(np.std(val,axis =0) +1e-9)

                # print("mel_featur", np.sum(mel_features))
                mel = normalise_acoustic_features(sample['mel_spec'].squeeze(),self.norm_stats_acoustic['mel_spec'])
                #mel = sample['mel_spec'].squeeze()
                acoustic_features[idx] = np.concatenate((mel,ft),axis = 1)
            else:
                 acoustic_features[idx]  = ft
            # 

        return acoustic_features,visual_features

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

