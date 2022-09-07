from fastai import *
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal
import librosa
import librosa.feature
import librosa.display
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
from pathlib import *
import os


sound_path = Path('sounds')
image_path = Path('specs')

# load data and turn it into spectrograms
def create_spectrograms():
    sound_types = ['human', 'not_human']
    for o in sound_types:
        dest = (sound_path/o)
        dest.mkdir(exist_ok=True)
        for filename in os.scandir(dest):
            # signal, sr = librosa.load(filename)
            # mfc = librosa.feature.melspectrogram(y=signal, sr=sr)
            # librosa.display.specshow(mfc, y_axis = None, x_axis = None)

            x, sr = librosa.load(filename, sr=44100)
    
            # stft is short time fourier transform
            X = librosa.stft(x)
            
            # convert the slices to amplitude
            Xdb = librosa.amplitude_to_db(abs(X))
            
            # ... and plot, magic!
            plt.figure(figsize=(14, 5))
            librosa.display.specshow(Xdb, sr = sr, x_axis = None, y_axis = None)
            plt.colorbar()
            fig=plt.Figure()

            pre, ext = os.path.splitext(filename)
            new_filename = pre + ".png"
            image_save_filename = os.path.join(image_path, new_filename)
            print(image_save_filename)
            with open(image_save_filename, 'w+') as f:
                fig.savefig(image_save_filename)


    # sample_rate, samples = wav.read(filename)
    # frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

    # plt.pcolormesh(times, frequencies, np.log(spectrogram))

    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')

    # plt.show()

# get path to images
sound_types = ['human', 'not_human']
path = Path('sounds')
if not path.exists():
    path.mkdir()
    for o in sound_types:
        dest = (path/o)
        dest.mkdir(exist_ok=True)

create_spectrograms()

# make a DataBlock, which is loaded into a dataloaders
# sounds = DataBlock(
#     blocks=(ImageBlock, CategoryBlock),
#     get_items = get_spec,
#     splitter=RandomSplitter(valid_pct=0.2, seed=42),
#     get_y=parent_label,
#     item_tfms=Resize(128))

# dls = sounds.dataloaders(path)
# dls.valid.show_batch(max_n=4, nrows=1)