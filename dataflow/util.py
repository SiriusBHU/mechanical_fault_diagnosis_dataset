"""
    Author: Sirius HU
    Created Date: 2019.10.10
    Function:
        signal_split_as_samples:
            sampling samples (a series of time slots with fixed sampling-points) from a long time sequence (signal),
            where the starting-point of each sample is selected randomly from the multiple points in the time sequence.

        convert_1d_to_chs1d
            given a set of multiple samples without channel-axis,
            add a 'channel' axis to each sample.
            the given data_set.shape = (nums, length)
            after this func, data_set.shape = (num, channel, length), where channel = 1

        convert_chs1d_to_1d
            the inverse func of convert_1d_to_chs1d

        onehot_encoding
            given a array of labels = [0, 1, 2, 1, 4, ..., 1] contain n classes labels
            encoding it in the form of one-hot

        decoding_onehot
            the inverse func of onehot_encoding

"""

import numpy as np
import random as rd


def signal_split_as_samples(signal_chs, sample_len, sample_num):

    """
        this function split vibration signals (in the time domain) with their [length] >> [sample_len]
        into multiple samples with the length of [sample_len]

        input :
            signals_chs: signals_chs.shape = (channels, length)
                         signals in different channels are synchronized
            sample_len: the length of expected samples
            sample_num: the number of expected samples

        return:
            samples: samples.shape = (sample_num, channels, sample_len)

        the beginning point of each sample are randomly chosen from all the points in the original signal
    """

    signal_chs = np.array(signal_chs)
    if len(signal_chs.shape) != 2:
        raise ValueError("expected signals_chs with 2 dims: (channels, length), but got %d dim(s)"
                         % len(signal_chs.shape))

    chs, length = signal_chs.shape

    if sample_num <= 0:
        raise ValueError("expected sample_num > 0, but got [sample_num: %d]" % sample_num)

    if sample_len <= 0:
        raise ValueError("expected sample_len > 0, but got [sample_len: %d]" % sample_len)

    if sample_len >= length:
        raise ValueError("expected sample_len < the length of signals, but got [sample_len: %d] > [length %d]"
                         % (sample_len, length))

    samples = []
    nn = length - sample_len
    for i in range(sample_num):
        n = rd.randint(0, nn)
        sample = signal_chs[:, n:n + sample_len].reshape(chs, sample_len)
        samples.append(sample)

    return np.array(samples)


def convert_1d_to_chs1d(data_no_chs, channels):

    """
        this function convert the data, whose channels have been flatten, into data_chs,
        with its channels being recovered

        input:
            data_no_chs: data whose channels have been flatten, data_no_chs.shape = (num, length)
            channels: channel number, we need this to fold the data to recover its channels

        return:
            data_chs: data_chs.shape = (num, chs, length // chs)
    """

    data_no_chs = np.array(data_no_chs)
    if len(data_no_chs.shape) != 2:
        raise ValueError("expected data with 2 dims: (number, length), but got %d dim(s)"
                         % len(data_no_chs.shape))

    num, length = data_no_chs.shape

    if channels <= 0:
        raise ValueError("expected channels > 0, but got [channels: %d]" % channels)

    if length % channels != 0:
        raise ValueError("expected channels should divided the length of data, "
                         "but got [length: %d] mod [channels: %d] !=0" % (length, channels))

    return data_no_chs.reshape(num, channels, int(length / channels))


def convert_chs1d_to_1d(data_chs):

    """
        this function flat channel-dimension of data

        input:
            data_chs: data with multiple channels, data_chs.shape = (num, channels, length)

        return:
            data_no_chs: data_no_chs.shape = (num, length * channels)
    """

    data_chs = np.array(data_chs)
    if len(data_chs.shape) != 3:
        raise ValueError("expected data with 3 dims: (number, channels, length), but got %d dim(s)"
                         % len(data_chs.shape))

    num, channels, length = data_chs.shape

    return data_chs.reshape(num, channels * length)


def onehot_encoding(labels, class_num=None):

    labels = np.array(labels)
    if len(labels.shape) == 1:
        labels.reshape(-1, 1)

    if len(labels.shape) != 2:
        raise ValueError("expected labels with 2 dims: (number, 1), but got %d dims"
                         % len(labels.shape))

    if np.shape(labels)[1] != 1:
        raise ValueError("expected label should be a scalar, but got a vector with %d elements"
                         % np.shape(labels)[1])

    if sum(labels.astype(np.int) - labels):
        raise ValueError("expected labels should all be integers, but existing decimal")

    if class_num != int(class_num):
        raise ValueError("expected class_num should be a integer, but got float %.f"
                         % class_num)

    if class_num < np.max(labels):
        raise ValueError("expected class_num should not smaller than max(labels) but got %d < %d"
                         % (class_num, np.max(labels)))

    labels = np.array(labels, dtype=np.int)
    num, _ = labels.shape
    labels_onehot = np.zeros((num, class_num))
    for i in range(num):
        labels_onehot[i, labels[i, 0]] = 1

    return labels_onehot


def decoding_onehot(labels_onehot):

    labels_onehot = np.array(labels_onehot)
    if len(labels_onehot.shape) != 2:
        raise ValueError("expected labels with 2 dims: (number, class_num), but got %d dims"
                         % len(labels_onehot.shape))

    return np.argmax(labels_onehot, axis=1)



