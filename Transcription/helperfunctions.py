import numpy as np
import os
import torch

from pypianoroll import Multitrack, Track, load, parse

def get_audio_segment(filepath,s, window):
    sz_float = 4 # size of a float
    epsilon = 10e-8 # fudge factor for normalization
    """
    Args:
        filepath (str): path to the bin file
        s (int): Start position of segment
        window (int): how many data points to get
    Returns:
        audio waveform
    """

    with open(filepath, 'rb') as f:
        f.seek(s*sz_float, os.SEEK_SET)
        x = np.fromfile(f, dtype=np.float32, count=int(window))

    x /= np.linalg.norm(x) + epsilon

    return x

def access_full(path):
    with open(path, 'rb') as f:
        x = np.fromfile(f, dtype=np.float32)
    return x

def get_piano_roll(filepath, model, device, window=16384, stride=1000, offset=44100, count=7500, batch_size=500, m=128):
    sf=4
    x = access_full(filepath)
    if stride == -1:
        stride = (x.shape[0] - offset - int(sf*window))/(count-1)
        stride = int(stride)
        print("Number of stride = ", stride)
    else:
        count = (x.shape[0]- offset - int(sf*window))/stride + 1
        count = int(count)
        
    X = np.zeros([count, window])
    Y = np.zeros([count, m])    
        
    for i in range(count):
        X[i,:] = get_audio_segment(filepath, offset+i*stride, window)
    with torch.no_grad():
        Y_pred = torch.zeros([count,m])
        for i in range(len(X)//batch_size):
            print(f"{i}/{(len(X)//batch_size)} batches", end = '\r')
            X_batch = torch.tensor(X[batch_size*i:batch_size*(i+1)]).float().to(device)
            Y_pred[batch_size*i:batch_size*(i+1)] = model(X_batch).cpu()
    
    return Y_pred

def export_midi(Y_pred, path):
    # Create a piano-roll matrix, where the first and second axes represent time
    # and pitch, respectively, and assign a C major chord to the piano-roll
    # Create a `pypianoroll.Track` instance
    track = Track(pianoroll=Y_pred*127, program=0, is_drum=False,
                  name='my awesome piano')   
    multitrack = Multitrack(tracks=[track], tempo=60, beat_resolution=86)
    multitrack.write(path)    