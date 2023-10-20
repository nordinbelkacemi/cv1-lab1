import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import os
import sys
from torchvision import transforms
import torchvision
from dataclasses import dataclass, InitVar
from sklearn.cluster import KMeans
import cv2
from typing import Callable, List
from skimage.feature import hog

@dataclass
class Config:
    mode: str
    vocab_size: int

    def __post_init__(self):
        if self.mode not in ["sift", "hog"]:
            raise ValueError("Mode must be 'sift' or 'hog'")

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def load_pickle(f):
    if sys.version_info[0] == 2:
        return pickle.load(f)
    elif sys.version_info[0] == 3:
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(sys.version))


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, "rb") as f:
        datadict = load_pickle(f)
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
        Y = np.array(Y)
        return X, Y


###  suggested reference: https://pytorch.org/tutorials/
# recipes/recipes/custom_dataset_transforms_loader.html?highlight=dataloader
# functions to show an image

class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, download=True, N=None, mode="sift"):
        """
                Initializes a CIFAR10_loader instance.

                Args:
                    root (str): Root directory of the CIFAR-10 dataset.
                    train (bool, optional): If True, loads the training data. If False, loads the test data. Defaults to True.
                    transform (callable, optional): A transform to apply to the data. Defaults to None.
                    N (int, optional): Maximum number of samples per class. Defaults to None.
        """
        super().__init__(root=root, train=train, transform=transform, download=download)
        self.mode = mode
        self.N = N
        self.data_update()

    def data_update(self):
        assert len(self.data) == len(self.targets)
        label_mapping = {0: 0, 2: 1, 8: 2, 7: 3, 1: 4}

        new_data = []
        new_targets = []
        class_counter = np.zeros(5)

        for item in range(len(self.data)):
            label = self.targets[item]
            if label in label_mapping:
                new_label_value = label_mapping[label]
                if self.N is None or class_counter[new_label_value] < self.N:
                    # Increment the class_counter and add the data and new label
                    if self.train:
                        if not self._are_descriptors_empty(self.data[item]):
                            class_counter[new_label_value] += 1
                            new_data.append(self.data[item])
                            new_targets.append(new_label_value)
                    else:
                        class_counter[new_label_value] += 1
                        new_data.append(self.data[item])
                        new_targets.append(new_label_value)
            if self.N is not None and np.all(class_counter == self.N):
                break

        self.data = np.asarray(new_data)
        self.targets = np.asarray(new_targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img, target = self.data[item], self.targets[item]

        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target
    
    def _are_descriptors_empty(self, img):
        """true if the descriptors of the image are empty"""
        if self.mode == "sift":
            return cv2.xfeatures2d.SIFT_create().detectAndCompute(img, None)[1] is None
        if self.mode == "hog":
            return False


def extract_keypoints_and_descriptors_sift(img):
    return cv2.xfeatures2d.SIFT_create().detectAndCompute(img, None)


def extract_descriptors_sift(img):
    return cv2.xfeatures2d.SIFT_create().detectAndCompute(img, None)[1]


def extract_descriptors_hog(img):
    histograms = hog(img, feature_vector=False)
    n_blocks_row, n_blocks_col, n_cells_row, n_cells_col, n_orient = histograms.shape
    return histograms.reshape(n_blocks_row * n_blocks_col, n_cells_row * n_cells_col * n_orient)


def extract_descriptors(imgs, extractor_fn):
    return [extractor_fn(img) for img in imgs]


def select_random_image_ids(labels, n):
    """ Randomly selects an equal amount of image ids from each class.
    Args:
        labels (np.array): 1D image labels array
        n (int): number of images per class
    Returns:
        selected_ids (dict): a dictionary that maps class labels to the selected ids (np.ndarray)
    """
    selected_ids = {}
    for img_class in np.unique(labels):
        selected_ids[img_class] = np.random.choice(np.where(labels == img_class)[0], n, replace=False)
    return selected_ids


def get_vocab_subset_size(labels, vocab_subset_ratio):
    n_classes = np.unique(labels).shape[0]
    return int((len(labels) * vocab_subset_ratio) // n_classes)


@dataclass
class Encoding:
    visual_words: np.ndarray    # (num_descriptors,) i-th element is the cluster (int) that the i-th descriptor belongs to
    histogram: np.ndarray       # (vocab_size,) normalized histogram


class Encoder:
    def __init__(self, training_data, training_labels, vocab_subset_ratio, vocab_size, mode) -> None:
        self.vocab_subset_ratio = vocab_subset_ratio
        self.vocab_size = vocab_size
        self.extractor_fn = {"sift": extract_descriptors_sift, "hog": extract_descriptors_hog}[mode]
        (
            self.kmeans,
            self.vocab_subset_ids,
            self.vocab_subset_descriptors
        ) = self._build_visual_vocab(
            training_data,
            training_labels,
        )

    def _build_visual_vocab(self, data, labels):
        """
        Builds a visual vocabulary by fitting kmeans clustering on image descriptors from SIFT.
        
        Args:
            data (np.array): array of training images with shape (n_images, height, width, n_channels)
            labels (np.array): class labels
        Returns:
            kmeans: sklearn KMeans object fitted to the data
            subset_descriptors (list): the descriptors of each image from the selected subset
        """
        n_imgs_per_class = get_vocab_subset_size(labels, self.vocab_subset_ratio)

        # Select the data and extract the descriptors
        selected_class_ids = select_random_image_ids(labels, n_imgs_per_class)
        selected_ids = np.array([v for v in selected_class_ids.values()]).flatten()
        vocab_subset = data[selected_ids]
        subset_descriptors = extract_descriptors(vocab_subset, self.extractor_fn)

        # Remove images with empty descriptors
        empty_descriptors_ids = [i for i, s in enumerate(subset_descriptors) if s is None]
        selected_ids = np.delete(selected_ids, empty_descriptors_ids)
        subset_descriptors = [s for s in subset_descriptors if s is not None]

        # Cluster descriptors
        kmeans = KMeans(n_clusters=self.vocab_size, random_state=42)
        kmeans.fit(np.vstack(subset_descriptors))

        return kmeans, selected_ids, subset_descriptors

    def encode_features(self, data):
        """
        Encodes the images by extracting the descriptors and assigning each of them
        to a cluster from kmeans (visual vocabulary)
        
        Args:
            data (np.array): data to encode
            kmeans (sklearn.cluster.KMeans): a kmeans object (corresponds to a visual vocabulary)
        
        Returns:
            idx_to_encoding (dict[int, np.array]): a dictionary where keys are the id of an image
                and values are the encoding. In the encoding, we have a list containing the clusters that each
                descriptor from the image belongs to, as well as a normalized histogram (counts relative
                occurrences of each visual word for a given image)
        """
        # Extract descriptors and keypoints from the whole dataset
        data_descriptors = extract_descriptors(data, self.extractor_fn)

        idx_to_encoding = {}
        for idx, descriptors in enumerate(data_descriptors):
            if descriptors is not None:
                visual_words = self.kmeans.predict(descriptors)
                histogram = np.histogram(visual_words, bins=np.arange(self.vocab_size + 1), density=True)[0]
                idx_to_encoding[idx] = Encoding(visual_words, histogram)
            else:
                idx_to_encoding[idx] = Encoding([], np.zeros(self.vocab_size))

        return idx_to_encoding


def f(i, sorted_gts):
    """
    Args:
        i (int): index (1-based, meaning that i=1 refers to the 1st element instead of i=0)
        sorted_gts (np.ndarray): ground truths sorted based on prediction scores

    Returns:
        int: number of 1s in the first i elements of sorted_gts
    """
    if sorted_gts[i - 1] == 1:  # i is 1-based
        return np.sum(sorted_gts[:i])
    else:
        return 0


def AP(preds, gts):
    """Calculates AP given predictions and ground truths for N test samples.
    Sorts predictions based on prediction score (which is the prediction value itself)
    and uses this sorted order to sort ground truths. The given formula for calculating
    AP is then applied.

    Args:
        preds (np.ndarray): predictions (1s and 0s) of shape (N,) 
        gts (np.ndarray): ground truths (1s and 0s) of shape (N,)
    Returns:
        float: Average precision, as defined in the question.
    """
    sorted_gts = gts[np.argsort(preds)]
    m_c = np.sum(gts)
    return 1 / m_c * np.sum([f(i, sorted_gts) / i for i in range(1, len(gts) + 1)])


def mAP(classification_results, labels):
    """Computes mAP on N test samples.

    Args:
        classification_results (dict[int, np.ndarray]): Dict where a key is a class index
            and the value is an array of predictions (1s and 0s) of shape (N,)
        labels (np.ndarray): Test labels

    Returns:
        float: mAP, i.e. the mean of APs for each class
    """
    return np.mean([
        AP(
            preds=predictions,
            gts=np.array(labels == class_id, dtype=np.int32)
        )
        for class_id, predictions in classification_results.items()
    ])


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = CIFAR10('./data', train=True, transform=transform, download=True, N=None)
    print(f"Train data: {train_set.data.shape}")
    print(f"Train labels: {train_set.targets.shape}")

    trainloader = DataLoader(train_set, batch_size=4,
                             shuffle=True, num_workers=2)
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    print(images.shape)

    test_set = CIFAR10('./data', train=False, transform=transform, download=True, N=None)
    print(f"Test data: {test_set.data.shape}")
    print(f"Test labels: {test_set.targets.shape}")
