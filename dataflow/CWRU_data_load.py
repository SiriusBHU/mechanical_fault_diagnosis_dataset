
"""
    Case Western Reserve University Bearing DataSet ReadFile
    Author: Sirius HU
    Created Date: 2018.10.31
    Modified_v1 Date: 2019.10.11

    Data Introduction:
        here we have 10 testing bearings, they are:
            among them, 1 are normal one
                        3 are inner race fault, with different fault diameters of 0.007, 0.014, 0.021 inches
                        3 are outer race fault, with different fault diameters of 0.007, 0.014, 0.021 inches
                        3 are ball fault, with different fault diameters of 0.007, 0.014, 0.021 inches

        all the faulty bearings are damaged manually by EDM (电火花加工).
        after selection， each one is tested under 4 different working conditions,
        and vibration signals are recoreded on both DE and BA.
        we select vibrations on DE for diagnostics, so here we totally have 40 testing cases,
        each cases contains one channel vibration signal.

    for more details, refer to:
        https://csegroups.case.edu/bearingdatacenter/pages/download-data-file
        Smith W A, Randall R B. Rolling element bearing diagnostics using the Case Western Reserve University
        data: A benchmark study[J]. Mechanical Systems and Signal Processing, 2015,64-65:100-131.

    Function:
"""

import logging
import os
import zipfile
import numpy as np
from six.moves import urllib
import dataflow.util as ut


class CaseWesternBearing(object):

    """
        using case:
        cwdata = CaseWesternBearing(sample_num, sample_len, path_project)
        # as soon as we declare the class CaseWesternBearing,
        # it will check if there has a prepared testing-case-level dataset
        # if not, it will generate it during __init__

        data_wc, labels_wc = cwdata.working_condition_transferring(few_num=100)
        data_ar, labels_ar = cwdata.fault_extent_transferring()
        # after __init__, we can generate transferring dataset, by two funcs:
        #       1. working_condition_transferring
        #       2. fault_extent_transferring
        # they will first check if the dataset has been loaded in the memory,
        # then check if we need to generate few-shot set (few_num=None or not),
        # and then form transferring set

    """
    # set processed dataset url
    __url = "http://github.com/TJUSIRIUS/mechanical_fault_diagnosis_dataset/dataset/Diagnostics/CWRU_data/CWRU_data.zip"

    def __init__(self,
                 sample_num=None, sample_len=None,
                 path_project=None,
                 path_txt=None, path_cache=None):
        """
            check store path of the Case Western Reserve University bearing data
            check the whether the 40-testing-case-level dataset exist
            if not, generate them

            note that: actually we generate 48 sub-sets instead of 40,
                       because in faulty extent transferring scenarios,
                       we need extra normal data to keep the balance of the training and testing set

                       when it comes to working condition transferring, only first 40 sub-sets will be used
        """

        self.path_txt = path_txt
        self.path_cache = path_cache
        self.path = path_project

        if not self.path:
            self.path = os.getcwd()

        if not self.path_txt:
            self.path_txt = self.path + "\\dataset\\Diagnostics\\CWRU_data\\"
        if not self.path_cache:
            self.path_cache = self.path + "\\datacache\\CWRU_dataset\\"

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
                logging.warning("no original .text files")
                self._download_data()

            if not sample_num or not sample_len:
                raise KeyError("keys [sample_num] and [sample_len] are needed for sampling")

            logging.info("sampling orignal .text files to formulate raw testing-case-level dataset")
            self.dataset_prepare_CWRU(sample_num, sample_len)

        logging.info("testing-case-level dataset has been prepared !")

        # prepare a interface to load testing-case-level dataset
        self.data, self.labels_wc, self.labels_fs = None, None, None

    def _download_data(self):
        """
            this function aims to download the processed CWRU Dataset,
            and unzip it
            after unzipping, delete the compressed file

            Note: the dataset is a set of processed *.txt files from original .mat files
        """
        # if no *.zip files, try to download
        files = os.listdir(self.path_txt)
        if "CWRU_data.zip" not in files:
            logging.info("starting download processed CWRU bearing data")
            url = self.__url
            logging.info("--> downloading from %s" % url)
            _data = urllib.request.urlopen(url)
            _file = url.strip().split(os.altsep)[-1]
            _file = os.path.join(self.path_txt, _file)
            with open(_file, "wb") as f:
                f.write(_data.read())
            logging.info("successfully download processed CWRU bearing data package")

        # now try to unzip the compressed *.zip files
        # after unzip, delete the *.zip files
        logging.info("unzip files ...")
        _file = "CWRU_data.zip"
        with zipfile.ZipFile(os.path.join(self.path_txt, _file), 'r') as _unzip_ref:
            _unzip_ref.extractall(self.path_txt)
        os.remove(_file)
        logging.info("successfully unzip CWRU bearing data")
        return

    def filename_CWRU(self):

        files = os.listdir(self.path_txt)
        label_wc = [1, 2, 3, 4, 5, 6, 7, 8, 9,
                    1, 2, 3, 4, 5, 6, 7, 8, 9,
                    1, 2, 3, 4, 5, 6, 7, 8, 9,
                    1, 2, 3, 4, 5, 6, 7, 8, 9,
                    0, 0, 0, 0]

        label_fs = [1, 1, 1, 2, 2, 2, 3, 3, 3,
                    1, 1, 1, 2, 2, 2, 3, 3, 3,
                    1, 1, 1, 2, 2, 2, 3, 3, 3,
                    1, 1, 1, 2, 2, 2, 3, 3, 3,
                    0, 0, 0, 0]

        return files, label_wc, label_fs

    def data_clean_CWRU(self):

        """
            here 40 signals stored in .text files
            this function reads data in each file, and convert them from str to float,
            and then packages them into one list
        """

        os.chdir(self.path_txt)
        files, filelabesl_wc, filelabels_fs = self.filename_CWRU()

        signals = []
        for file in files:
            try:
                with open(file, "r") as fr:
                    signal = fr.readlines()
                    signal = np.array([float(item.strip()) for item in signal])
            except IOError:
                raise IOError("some wrong with the file %s" % file)
            signals.append(signal.reshape(1, -1))

        os.chdir(self.path)
        return signals, filelabesl_wc, filelabels_fs

    def dataset_prepare_CWRU(self, sample_num, sample_len):

        """
            this function first use [data_clean_CWRU] to read signals from .text file
            then samples examples (.shape = sample_num, channels, sample_len) from each signal
            40 (actually 48, last 8 is repeated sampled from normal cases) sub-sets will be generate
            and then be packaged into one dataset.

            this dataset will be used for 2 transferring scenarios, including:
                working condition transferring
                fault extent transferring
        """

        logging.info("generating new dataset with each case has (sample_num = %d, sample_len = %d)"
                     % (sample_num, sample_len))

        signals, filelabels_wc, filelabels_fs = self.data_clean_CWRU()
        data, labels_wc, labels_fs = [], [], []
        for k, signal in enumerate(signals):
            data.append(ut.signal_split_as_samples(signal, sample_len, sample_num))
            sample_labels_wc = np.ones((sample_num, 1)) * filelabels_wc[k]
            sample_labels_wc = ut.onehot_encoding(sample_labels_wc, class_num=10)
            labels_wc.append(sample_labels_wc)

            sample_labels_fs = np.ones((sample_num, 1)) * filelabels_fs[k]
            sample_labels_fs = ut.onehot_encoding(sample_labels_fs, class_num=4)
            labels_fs.append(sample_labels_fs)

        for i in range(2):
            for signal in signals[-4:]:
                data.append(ut.signal_split_as_samples(signal, sample_len, sample_num))
                sample_labels_wc = np.zeros((sample_num, 1))
                sample_labels_wc = ut.onehot_encoding(sample_labels_wc, class_num=10)
                labels_wc.append(sample_labels_wc)

                sample_labels_fs = np.zeros((sample_num, 1))
                sample_labels_fs = ut.onehot_encoding(sample_labels_fs, class_num=4)
                labels_fs.append(sample_labels_fs)

        data, labels_wc, labels_fs = np.array(data), np.array(labels_wc), np.array(labels_fs)
        os.chdir(self.path_cache)
        np.savez("CWRU_1D_signal.npz", data=data, labels_wc=labels_wc, labels_fs=labels_fs)
        os.chdir(self.path)

        logging.info("new dataset has been stored")

        return 0

    def dataset_read_CWRU(self):

        dataset = np.load(self.path_cache + "\\CWRU_1D_signal.npz")
        self.data = dataset["data"]
        self.labels_wc = dataset["labels_wc"]
        self.labels_fs = dataset["labels_fs"]

    def working_condition_transferring(self, few_num=None):

        if not isinstance(self.data, np.ndarray):
            logging.info("loading testing-case-level dataset into memory")
            self.dataset_read_CWRU()
            logging.info("testing-case-level dataset loaded !")

        data, labels = self.data[:40], self.labels_wc[:40]

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

        wc_seq = [[ 0,  1,  2,  3,  4,  5,  6,  7,  8, 36],
                  [ 9, 10, 11, 12, 13, 14, 15, 16, 17, 37],
                  [18, 19, 20, 21, 22, 23, 24, 25, 26, 38],
                  [27, 28, 29, 30, 31, 32, 33, 34, 35, 39]]

        data = [[data[i] for i in sub_seq] for sub_seq in wc_seq]
        labels = [[labels[i] for i in sub_seq] for sub_seq in wc_seq]
        data, labels = np.array(data), np.array(labels)
        data, labels = data.reshape(4, 10 * num, channels, length), labels.reshape(4, 10 * num, -1)
        logging.info("working condition transferring set prepared, with [few-shot-num: %s]" % str(few_num))

        return data, labels

    def fault_extent_transferring(self, few_num=None):

        if not isinstance(self.data, np.ndarray):
            logging.info("loading testing-case-level dataset into memory")
            self.dataset_read_CWRU()
            logging.info("testing-case-level dataset loaded !")

        data, labels = self.data, self.labels_fs

        if len(data.shape) != 4:
            raise ValueError("excepted data.shape = (testing_cases, working_conditions, num, channels, length), "
                             "but got %d dim(s)" % len(data.shape))

        if len(labels.shape) != 3:
            raise ValueError("excepted labels.shape = (testing_cases, working_conditions, num, class_num), "
                             "but got %d dim(s)" % len(labels.shape))

        class_wcs, num, channels, length = data.shape
        if few_num:
            if few_num < 0 or few_num > num:
                raise ValueError("expected 0 < few-num < total sampling number (%d), but got %d" % (num, few_num))
            logging.info("according to few-shot sample number, extracting few-shot set")
            perm = np.arange(num)
            np.random.shuffle(perm)
            data, labels = data[:, perm[:few_num]], labels[:, perm[:few_num], :]
            num = few_num

        fs_seq = [[0,  9, 18, 27, 3, 12, 21, 30, 6, 15, 24, 33, 36, 37, 38, 39],
                  [1, 10, 19, 28, 4, 13, 22, 31, 7, 16, 25, 34, 40, 41, 42, 43],
                  [2, 11, 20, 29, 5, 14, 23, 32, 8, 17, 26, 35, 44, 45, 46, 47]]

        data = [[data[i] for i in sub_seq] for sub_seq in fs_seq]
        labels = [[labels[i] for i in sub_seq] for sub_seq in fs_seq]
        data, labels = np.array(data), np.array(labels)
        data, labels = data.reshape(3, 16 * num, channels, length), labels.reshape(3, 16 * num, -1)

        logging.info("fault extent transferring set prepared, with [few-shot-num: %s]" % str(few_num))

        return data, labels


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s--%(name)s--%(module)s--%(levelname)s]: %(message)s")

    " fix me "
    path_project = "E:\\mechanical_fault_diagnosis_dataset\\"
    sample_num, sample_len = 500, 2048

    cwdata = CaseWesternBearing(sample_num, sample_len, path_project=path_project)
    cwdata.dataset_prepare_CWRU(sample_num, sample_len)
    data_wc, labels_wc = cwdata.working_condition_transferring()
    data_ar, labels_ar = cwdata.fault_extent_transferring()

    print(1)
