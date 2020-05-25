"""
    Society of MFPT (Machinery Failure Prevention Technology) Bearing Dataset
    Author: Sirius HU
    Created Date: 2018.08.27
    Modified-v1: 2019.10.12

    Data Introduction:

        here we have 3 testing bearings (1 normal, 1 inner race fault, 1 outer race fault),

        after selection， the normal bearing is tested under 1 working condition for 3 times,
                                    sampling-rate=97656 Hz  (270 lbs, 1500 rpm)
                          the outer-faulty bearing is tested under 6 working conditons for 1 time
                                    sampling-rate=48828 Hz         (50, 100, 150, 200, 250 and 300 lbs, 1500 rpm)
                          the inner-faulty bearing is tested under 6 working conditons for 1 time
                                    sampling-rate=48828 Hz         (50, 100, 150, 200, 250 and 300 lbs, 1500 rpm)
        so there have totally 15 testing cases, each cases contains one channel vibration signal.

    for more details, refer to:
        https://mfpt.org/fault-data-sets/

    Functions:

"""

import os
import logging
import numpy as np
import dataflow.util as ut


class MFPTBearing(object):

    """
        using case:
            mfptdata = MFPTBearing(sample_num, sample_len, path_project=path_project)
            # as soon as we declare the class MFPTBearing,
            # it will check if there has a prepared testing-case-level dataset
            # if not, it will generate it during __init__

            mfptdata.dataset_prepare_MFPT(sample_num, sample_len)
            # if we want to generate a new dataset with different sample_num and sample_len, we can use func:
            #       dataset_prepare_MFPT

            data, labels = mfptdata.working_condition_transferring()
            # after __init__, we can generate transferring dataset, by func:
            #       working_condition_transferring
            # it will first check if the dataset has been loaded in the memory,
            # then check if we need to generate few-shot set (few_num=None or not),
            # and then form transferring set

    """

    def __init__(self,
                 sample_num=None, sample_len=None,
                 path_project=None,
                 path_txt=None, path_cache=None):
        """
            check store path of the Machinery Failure Prevention Technologybearing data
            check the whether the 15-testing-case-level dataset exist
            if not, generate them

            note that: actually we generate 18 sub-sets instead of 15,
                       because in the transferring scenario,
                       we need extra normal data to keep the balance of the training and testing set
        """

        self.path_txt = path_txt
        self.path_cache = path_cache
        self.path = path_project

        if not self.path:
            self.path = os.getcwd()

        if not self.path_txt:
            self.path_txt = self.path + "\\dataset\\Diagnostics\\MFPT_data\\"
        if not self.path_cache:
            self.path_cache = self.path + "\\datacache\\MFPT_dataset\\"

        # check dataset related directories  do or not exist
        if not os.path.exists(self.path_txt):
            logging.warning("no original .text path")
            os.makedirs(self.path_txt)

        if not os.path.exists(self.path_cache):
            logging.warning("no formulated dataset path")
            os.makedirs(self.path_cache)

        # check the dataset has or hasn't been prepared
        # if not, generate testing-case-level dataset
        if len(os.listdir(self.path_cache)) == 0:
            logging.warning("no formulated dataset")

            if not len(os.listdir(self.path_txt)):
                raise FileExistsError("no original .text files")

            if not sample_num or not sample_len:
                raise KeyError("keys [sample_num] and [sample_len] are needed for sampling")

            logging.info("sampling orignal .text files to formulate raw testing-case-level dataset")
            self.dataset_prepare_MFPT(sample_num, sample_len)

        logging.info("testing-case-level dataset has been prepared !")

        # prepare a interface to load testing-case-level dataset
        self.data, self.labels = None, None

    def filename_MFPT(self):

        files = os.listdir(self.path_txt)
        labels = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
                  0, 0, 0]

        return files, labels

    def data_clean_MFPT(self):

        """
            here 15 signals stored in .text files
            this function reads data in each file, and convert them from str to float,
            and then packages them into one list
        """

        os.chdir(self.path_txt)
        files, filelabels = self.filename_MFPT()

        signals = []
        for file in files:
            try:
                with open(file, "r") as fr:
                    signal = fr.readlines()
                    signal = np.array([float(item.strip()) for item in signal])
            except IOError:
                raise IOError("some wrong with the file %s" % file)
            if "normal" in file:
                signal = signal[::2]
            signals.append(signal.reshape(1, -1))

        os.chdir(self.path)
        return signals, filelabels

    def dataset_prepare_MFPT(self, sample_num, sample_len):

        """
            this function first use [data_clean_MFPT] to read signals from .txt file
            then samples examples (.shape = sample_num, channels, sample_len) from each signal
            15 (actually 18, last 3 is repeated sampled from normal cases) sub-sets will be generate
            and then be packaged into one dataset.

            this dataset will be used for 1 transferring scenario:
                working condition transferring
        """

        logging.info("generating new dataset with each case has (sample_num = %d, sample_len = %d)"
                     % (sample_num, sample_len))

        signals, filelabels = self.data_clean_MFPT()
        data, labels, = [], []
        for k, signal in enumerate(signals):
            data.append(ut.signal_split_as_samples(signal, sample_len, sample_num))
            sample_labels_wc = np.ones((sample_num, 1)) * filelabels[k]
            sample_labels_wc = ut.onehot_encoding(sample_labels_wc, class_num=3)
            labels.append(sample_labels_wc)

        for signal in signals[-3:]:
            data.append(ut.signal_split_as_samples(signal, sample_len, sample_num))
            sample_labels_wc = np.zeros((sample_num, 1))
            sample_labels_wc = ut.onehot_encoding(sample_labels_wc, class_num=3)
            labels.append(sample_labels_wc)

        data, labels = np.array(data), np.array(labels)
        os.chdir(self.path_cache)
        np.savez("MFPT_1D_Bearing_DataSet.npz", data=data, labels=labels)
        os.chdir(self.path)

        logging.info("new dataset has been stored")

        return 0

    def dataset_read_MFPT(self):

        dataset = np.load(self.path_cache + "\\MFPT_1D_Bearing_DataSet.npz")
        self.data = dataset["data"]
        self.labels = dataset["labels"]

    def working_condition_transferring(self, few_num=None):

        if not isinstance(self.data, np.ndarray):
            logging.info("loading testing-case-level dataset into memory")
            self.dataset_read_MFPT()
            logging.info("testing-case-level dataset loaded !")

        data, labels = self.data, self.labels

        if len(data.shape) != 4:
            raise ValueError("excepted data.shape = (testing_cases, working_conditions, num, channels, length), "
                             "but got %d dim(s)" % len(data.shape))

        if len(labels.shape) != 3:
            raise ValueError("excepted labels.shape = (testing_cases, working_conditions, num, class_num), "
                             "but got %d dim(s)" % len(labels.shape))

        cases, num, channels, length = data.shape
        if few_num:
            if few_num < 0 or few_num > num:
                raise ValueError("expected 0 < few-num < total sampling number (%d), but got %d" % (num, few_num))
            logging.info("according to few-shot sample number, extracting few-shot set")
            perm = np.arange(num)
            np.random.shuffle(perm)
            data, labels = data[:, perm[:few_num]], labels[:, perm[:few_num], :]
            num = few_num

        wc_seq = [[12,  0,  1],
                  [13,  2,  3],
                  [14,  4,  5],
                  [15,  6,  7],
                  [16,  8,  9],
                  [17, 10, 11]]

        data = [[data[i] for i in sub_seq] for sub_seq in wc_seq]
        labels = [[labels[i] for i in sub_seq] for sub_seq in wc_seq]
        data, labels = np.array(data), np.array(labels)
        data, labels = data.reshape(6, 3 * num, channels, length), labels.reshape(6, 3 * num, -1)
        logging.info("working condition transferring set prepared, with [few-shot-num: %s]" % str(few_num))

        return data, labels


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s--%(name)s--%(module)s--%(levelname)s]: %(message)s")

    " fix me "
    path_project = "D:\\DataSet_Preparation\\"
    sample_num, sample_len = 500, 2048

    mfptdata = MFPTBearing(sample_num, sample_len, path_project=path_project)
    mfptdata.dataset_prepare_MFPT(sample_num, sample_len)
    data, labels = mfptdata.working_condition_transferring()

    print(1)


