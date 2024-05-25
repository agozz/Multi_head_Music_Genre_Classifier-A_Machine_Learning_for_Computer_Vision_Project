import pandas as pd
import librosa.feature
import librosa.display as lplt
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from colorama import Fore, Style

gtzan_genres = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]


def create_annotations(create=True, saving_root='./GTZAN', lists=None):
    """
    Generate three .csv files containing the name of data, and correspondent label, for training, validation and test
    :param create: flag to allow the creation of annotations
    :param saving_root: folder root in which annotations are saved
    :param lists: contains the 3 lists with the names of data for training, validation and test
    """
    if create:
        print(f"{Fore.MAGENTA}Creating annotations{Style.RESET_ALL}")

        names = []
        labels = []
        for filename in tqdm(os.listdir('./GTZAN/images_3sec/chromas/')):
            names.append(filename)

            split = filename.split('.')
            labels.append(split[0])

        annotations = pd.DataFrame(list(zip(names, labels)), columns=['filenames', 'labels'])

        annotations[annotations['filenames'].isin(lists[0])].to_csv(saving_root + '/annotations_train_cerberus.csv')
        annotations[annotations['filenames'].isin(lists[1])].to_csv(saving_root + '/annotations_val_cerberus.csv')
        annotations[annotations['filenames'].isin(lists[2])].to_csv(saving_root + '/annotations_test_cerberus.csv')


def create_images(create=True):
    """
    Create three image representations for each audio track, In particular, from a single track, 10 images for every
    kind of depictions are obtained.
    :param create: flag to allow the creation of images
    """
    if create:
        os.makedirs('./GTZAN/images_3sec/chromas', exist_ok=True)
        os.makedirs('./GTZAN/images_3sec/mel_spec', exist_ok=True)
        os.makedirs('./GTZAN/images_3sec/mfcc', exist_ok=True)

        for genre in gtzan_genres:
            print(f"{Fore.MAGENTA}Creating images for:{Style.RESET_ALL} {genre}")

            for filename in tqdm(os.listdir('./GTZAN/genres_original/' + genre + '/')):
                audio_path = './GTZAN/genres_original/' + genre + '/' + filename
                split = filename.split('.')

                for i in range(10):
                    audio_data, sr = librosa.load(audio_path, offset=i * 3, duration=3)

                    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
                    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr)
                    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=20)

                    lplt.specshow(chroma, sr=sr)
                    plt.savefig('./GTZAN/images_3sec/chromas/' + split[0] + '.' + split[1] + '.' + str(i) + '.jpg',
                                bbox_inches='tight', pad_inches=0)
                    plt.clf()

                    lplt.specshow(librosa.power_to_db(mel_spec, ref=np.max), sr=sr)
                    plt.savefig('./GTZAN/images_3sec/mel_spec/' + split[0] + '.' + split[1] + '.' + str(i) + '.jpg',
                                bbox_inches='tight', pad_inches=0)
                    plt.clf()

                    lplt.specshow(mfcc, sr=sr)
                    plt.savefig('./GTZAN/images_3sec/mfcc/' + split[0] + '.' + split[1] + '.' + str(i) + '.jpg',
                                bbox_inches='tight', pad_inches=0)
                    plt.clf()
