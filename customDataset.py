import os
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
from utils import gtzan_genres
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class CustomGTZAN(Dataset):
    def __init__(self, csv_file, root_dir='./GTZAN/images_3sec', transform=None):
        """
        Create a custom version of the GTZAN dataset.
        :param csv_file: path for the annotations of the data
        :param root_dir: path for the images
        :param transform: list of transformation to be applied to the images
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.le = LabelEncoder().fit(gtzan_genres)
        self.encoding = self.le.transform(gtzan_genres)
        self.ohe = OneHotEncoder(handle_unknown='ignore').fit_transform(self.encoding.reshape(-1, 1)).toarray()

    def __len__(self):
        """
        :return: length of the set
        """
        return len(self.annotations)

    def __getitem__(self, index):
        """
        :param index: position of the object in the set
        :return: a set of images and the correspondent label
        """
        img_path1 = os.path.join(self.root_dir, 'chromas', self.annotations.iloc[index, 1])
        img_path2 = os.path.join(self.root_dir, 'mel_spec', self.annotations.iloc[index, 1])
        img_path3 = os.path.join(self.root_dir, 'mfcc', self.annotations.iloc[index, 1])

        image1 = io.imread(img_path1)
        image2 = io.imread(img_path2)
        image3 = io.imread(img_path3)

        label = self.annotations.iloc[index, 2]
        label = self.le.transform([label])
        label = self.ohe[:, label]

        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            image3 = self.transform(image3)

        return image1, image2, image3, label


def get_datasets(load_train=True, load_val=True, load_test=True, transform=None):
    """
    :param load_train: flag to allow the creation of the training set
    :param load_val: flag to allow the creation of the validation set
    :param load_test: flag to allow the creation of the test set
    :param transform: dict of transformations to be applied and divided according to training, validation and test
    :return: training, validation and test set
    """
    if load_train:
        if transform is None:
            train_dataset = CustomGTZAN(csv_file='./GTZAN/annotations_train_cerberus.csv', transform=None)
        else:
            train_dataset = CustomGTZAN(csv_file='./GTZAN/annotations_train_cerberus.csv', transform=transform['train'])

    else:
        train_dataset = None

    if load_val:
        if transform is None:
            val_dataset = CustomGTZAN(csv_file='./GTZAN/annotations_val_cerberus.csv', transform=None)
        else:
            val_dataset = CustomGTZAN(csv_file='./GTZAN/annotations_val_cerberus.csv', transform=transform['val_test'])
    else:
        val_dataset = None

    if load_test:
        if transform is None:
            test_dataset = CustomGTZAN(csv_file='./GTZAN/annotations_test_cerberus.csv', transform=None)
        else:
            test_dataset = CustomGTZAN(csv_file='./GTZAN/annotations_test_cerberus.csv',
                                       transform=transform['val_test'])
    else:
        test_dataset = None

    return train_dataset, val_dataset, test_dataset
