import numpy as np
import os

def normalise_acoustic_features(features,stats):

    mean = stats['mean']
    std = stats['std']
    
    return (features- mean)/std