"""
This module is used to download, extract, load, and delete the MNIST dataset.
"""
import os
import shutil
import urllib.request
import gzip
import numpy as np


def download_and_extract(verbose=True):
    """
    Download the MNIST data set and extract it into the folder ./data/MNIST

    :param verbose: Toggles verbose printing
    """
    dest = './data/MNIST'
    url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
    filename = url.split('/')[-1]
    filepath = os.path.join(dest, filename)
    if not os.path.exists(dest):
        if not os.path.exists(filepath):
            os.makedirs(dest)

            def progress(count, block_size, total_size):
                """
                Prints a progress bar for the MNIST download
                """
                perc = float(count * block_size) / float(total_size)
                width = 80 - len(">  Downloading {} {:.1f}%".format(
                    filename, perc * 100.0))
                if verbose:
                    print(
                        "\r>> Downloading {} [{}] {:.1f}%".format(
                            filename,
                            ('=' * int(perc * width)) + ">" + (' ' * int(
                                (1.0 - perc) * width)), perc * 100.0),
                        end='')

            filepath, _ = urllib.request.urlretrieve(url, filepath, progress)
            if verbose:
                print()
            statinfo = os.stat(filepath)
            if verbose:
                print("   Successfully downloaded", filename, statinfo.st_size,
                      'bytes.')
    extract_dir = os.path.join(dest, 'mnist.pkl')
    if not os.path.exists(extract_dir):
        if verbose:
            print(">> Extracting {}".format(filepath))
        with gzip.open(filepath, 'rb') as binary_data:
            with open(extract_dir, 'wb') as binary_out:
                shutil.copyfileobj(binary_data, binary_out)
        # tarfile.open(filepath, 'r:gz').extractall(dest)
        if verbose:
            print("   Successfully extracted {}".format(filepath))


def delete(verbose=True):
    """
    Deletes all downloaded data

    :param verbose: Toggles verbose printing
    """
    dest = './data/MNIST'
    if os.path.exists(dest):
        if verbose:
            print(">> Deleting {}".format(dest))
        shutil.rmtree(dest)
        if verbose:
            print("   Successfully deleted {}".format(dest))

def unpickle(file):
    """
    Opens and reads the binary MNIST batch data

    :param file: Path to file to unpickle
    :returns: Data from pickeled file
    """
    import pickle
    with open(file, 'rb') as binary:
        vals = pickle.load(binary, encoding='bytes')
        return vals
    return None

def load(file, verbose=True):
    """
    Loads a MNIST data file, and parses the binary into two numpy.ndarrays.

    :param file: Specifies validation or training or testing data set to load.
    :param verbose: Toggles verbose printing.
    :returns: A pair of data, the first is the input data, and the second is the matching labels.
    """
    download_and_extract(verbose)
    abs_file = os.path.abspath('./data/MNIST/mnist.pkl')
    data_set = 0
    if isinstance(file, int):
        data_set = file
    else:
        if "validation".startswith(file):
            data_set = 1
        elif "testing".startswith(file):
            data_set = 2
        else:
            data_set = 0
    train, valid, test = unpickle(abs_file)
    x_data = None
    y_data = None
    if data_set == 0:
        x_data, y_raw = train
    elif data_set == 1:
        x_data, y_raw = valid
    elif data_set == 2:
        x_data, y_raw = test
    y_data = np.zeros((10, np.size(x_data, 0)))
    for i in range(np.size(x_data, 0)):
        y_data[y_raw[i], i] = 1.0
    return x_data, y_data.T

def load_all(verbose=True):
    """
    Loads the MNIST data set, splitting into training and testing data sets.
    :param verbose: Toggles verbose printing.
    :returns: Four sets of data, the first is the input training data, and the
              second is the matching training labels, then the third is the
              input testing data, and the fourth is the output testing labels.
    """
    download_and_extract(verbose)
    source_dir = './data/MNIST/mnist.pkl'
    train, valid, test = unpickle(source_dir)
    x_data = np.concatenate([train[0], valid[0]])
    y_raw = np.concatenate([train[1], valid[1]])
    y_data = np.zeros((10, np.size(x_data, 0)))
    for i in range(np.size(x_data, 0)):
        y_data[y_raw[i], i] = 1.0
    x_test, y_raw = test
    y_test = np.zeros((10, np.size(x_test, 0)))
    for i in range(np.size(x_test, 0)):
        y_test[y_raw[i], i] = 1.0
    return x_data, y_data.T, x_test, y_test.T
