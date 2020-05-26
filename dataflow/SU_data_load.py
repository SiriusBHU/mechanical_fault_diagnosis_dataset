
"""
    Southeast University Bearing-Gear Dataset ReadFile
    Author: Sirius HU
    Created Date: 2018.10.31
    Modified-v1: 2019.10.11

    Data Introduction:
        here we have 10 testing cases, they are:
            5 bearings: health                      (1),
                        ball fault                  (1),
                        inner fault                 (1),
                        outer fault                 (1),
                        inner-outer combined fault  (1)

            5 gears: health         (1)
                     chipped tooth  (1)
                     Miss tooth     (1)
                     root           (1)
                     surface        (1)

        after selection， each one is tested under 2 different working conditions,
        and vibration signals are recoreded on 8 channels, including:
             1      -motor vibration,
             2,3,4  -vibration of planetary gearbox in three directions: x, y, and z,
             5      -motor torque,
             6,7,8  -vibration of parallel gear box in three directions: x, y, and z.

    for more details, refer to:
        https://github.com/cathysiyu/Mechanical-datasets
        S. S, S. M, R. Y, et al. Highly Accurate Machine Fault Diagnosis Using Deep Transfer
        Learning[J]. IEEE Transactions on Industrial Informatics, 2019,15(4):2446-2455.

    Function:
"""


import logging
import os
import numpy as np
import pandas as pd
import dataflow.util as ut


class SoutheastBearingGear(object):

    """
        using case:
            sudata = SoutheastBearingGear(sample_num, sample_len, path_project)

            # as soon as we declare the class SoutheastBearingGear,
            # it will check if there has 2 prepared testing-case-level datasets
            # if not, it will generate it during __init__


            data_bearing, labels_bearing = sudata.bearing_working_condition_transferring(few_num=60, chs=(1, 2, 3))
            data_gear, labels_gear = sudata.gear_working_condition_transferring(few_num=30, chs=3)

            # after __init__, we can generate transferring dataset of bearing and gear, by two funcs:
            #       1. bearing_working_condition_transferring
            #       2. gear_working_condition_transferring
            # they will first check if the dataset has been loaded in the memory,
            # then check if we need to generate few-shot set (few_num=None or not),
            # then check if we need to extract (a) certain channel(s) of the testing-case-dataset
            # and then form transferring set

    """
    # set dataset url
    __url = "https://github.com/cathysiyu/Mechanical-datasets/archive/master.zip"

    def __init__(self,
                 sample_num=None, sample_len=None,
                 path_project=None,
                 path_csv=None, path_npz=None, path_cache=None):

        """
            check store path of the Paderborn bearing data
            check the whether the 32 testing-case-level datasets exist
            if not, generate them
        """

        self.path_csv = path_csv
        self.path_npz = path_npz
        self.path_cache = path_cache
        self.path = path_project

        if not self.path:
            self.path = os.getcwd()

        if not self.path_csv:
            self.path_csv = self.path + "\\dataset\\Diagnostics\\SU_data\\"
        if not self.path_cache:
            self.path_cache = self.path + "\\datacache\\SU_dataset\\"
        if not self.path_npz:
            self.path_npz = self.path + "\\datacache\\SU_dataset\\_origin\\"

        # check dataset related directories  do or not exist
        if not os.path.exists(self.path_csv):
            logging.warning("no original .mat path")
            os.makedirs(self.path_csv)

        if not os.path.exists(self.path_cache):
            logging.warning("no formulated dataset path")
            os.makedirs(self.path_cache)

        if not os.path.exists(self.path_npz):
            logging.warning("no original .npz path")
            os.makedirs(self.path_npz)

        # check the dataset has or hasn't been prepared
        # if not, generate testing-case-level dataset
        if len(os.listdir(self.path_cache)) < 3:
            logging.warning("no formulated dataset")

            if len(os.listdir(self.path_npz)) < 20:
                logging.warning("no original .npz files")

                if not len(os.listdir(self.path_csv)):
                    raise FileExistsError("no original .csv files,\n"
                                          "please download from url: %s\n,"
                                          "unzip it add extract the 'bearingset' and 'gearset' "
                                          "directories in dir: \n%s"
                                          % (self.__url, self.path_csv))

                logging.info("convert orignal .mat files to original .npz files")
                self.origin_mat2npz_SU()
                logging.info(".npz files has been generated !")

            if not sample_num or not sample_len:
                raise KeyError("keys [sample_num] and [sample_len] are needed for sampling")

            logging.info("sampling orignal .npz files to formulate raw dataset with testing cases level dataset")
            self.dataset_prepare_SU(sample_num, sample_len)

        logging.info("testing-case-level dataset has been prepared !")

        # prepare a interface to load testing-case-level dataset
        self.data, self.labels, self.status = None, None, None

    def origin_mat2npz_SU(self):


        """
            here we aim to extract the vibration signals in .csv files,
            and restore them in .npy files for each bearing (total 20),
            each one constructed .npy file contains 8 channels vibration signals
        """

        sub_dirs = os.listdir(self.path_csv)
        if "bearingset" not in sub_dirs or "gearset" not in sub_dirs:
            raise ValueError("expected bearing set and / or gear set not in directory: %s" % self.path_csv)

        dirs = [self.path_csv + "\\bearingset\\", self.path_csv + "\\gearset\\"]
        for k, sub_dir in enumerate(dirs):

            files = os.listdir(sub_dir)
            for file in files:

                # 在处理混合精度数据时，low_momery 设置为 false，在处理同精度数据时，low_memory 设置为 True
                df = pd.read_csv(sub_dir + file, low_memory=False)
                values = df.values
                if len(values[16]) == 1:
                    values = values[12:].reshape(-1)
                    values = [item.strip().split("\t") for item in values]
                    values = np.array(values).astype(np.float)
                else:
                    values = values[15:, :8].astype(np.float)

                # 将 channels 这个维度放在前面
                values = values.transpose(1, 0)

                os.chdir(self.path_npz)
                if k == 0:
                    np.savez("Bearing_" + file[:-4] + ".npz", arr=values)
                else:
                    np.savez("Gear_" + file[:-4] + ".npz", arr=values)
        os.chdir(self.path)

    def filename_SU(self):

        filenames = os.listdir(self.path_npz)
        filenames = [filenames[:10], filenames[10:]]

        labels = [[1, 1, 4, 4, 0, 0, 2, 2, 3, 3],
                  [1, 1, 0, 0, 2, 2, 3, 3, 4, 4]]

        return filenames, labels

    def dataset_prepare_SU(self, sample_num, sample_len):

        """
            here 20 testing-case-level sub-sets will be generated and grepped into two datasets, including:
                    1. SU_1D_Bearing_DataSet.npz
                    2. SU_1D_Gear_Dataset.npz
            the 2 datasets will be prepared for one kind of transferring scenario:
                working condition transferring
        """

        logging.info("generating new dataset with each case has (sample_num = %d, sample_len = %d)"
                     % (sample_num, sample_len))

        filenames, filelabels = self.filename_SU()
        os.chdir(self.path_npz)
        data, labels = [[], []], [[], []]

        for k, files in enumerate(filenames):

            for l, file in enumerate(files):
                signal = np.load(file)["arr"]
                if len(signal.shape) != 2:
                    raise ValueError("excepted signal has 2 dims, but got %d dim(s)" % len(signal.shape))

                if np.shape(signal)[0] != 8:
                    raise ValueError("excepted signal's channels' number is 8 , but got %d" % np.shape(signal)[0])

                samples = ut.signal_split_as_samples(signal, sample_len, sample_num)
                data[k].append(samples)
                sample_labels = np.ones((sample_num, 1)) * filelabels[k][l]
                sample_labels = ut.onehot_encoding(sample_labels, 5)
                labels[k].append(sample_labels)

        data_bearing, data_gear = np.array(data[0]), np.array(data[1])
        labels_bearing, labels_gear = np.array(labels[0]), np.array(labels[1])

        os.chdir(self.path_cache)
        np.savez("SU_1D_Bearing_DataSet.npz", data=data_bearing, labels=labels_bearing)
        np.savez("SU_1D_Gear_DataSet.npz", data=data_gear, labels=labels_gear)
        os.chdir(self.path)

        logging.info("new dataset has been stored")
        return 0

    def dataset_read_SU(self, data_need="bearing"):

        if data_need == "bearing":
            dataset = np.load(self.path_cache + "\\SU_1D_Bearing_DataSet.npz")
            self.status = "bearing"
        else:
            dataset = np.load(self.path_cache + "\\SU_1D_Gear_DataSet.npz")
            self.status = "gear"

        self.data = dataset["data"]
        self.labels = dataset["labels"]

    def bearing_working_condition_transferring(self, few_num=None, chs=None):

        if not isinstance(self.data, np.ndarray) or self.status != "bering":
            logging.info("loading testing-case-level bearing dataset into memory")
            self.dataset_read_SU(data_need="bearing")
            logging.info("testing-case-level bearing dataset loaded !")

        data, labels = self.data, self.labels

        if len(data.shape) != 4:
            raise ValueError("excepted data.shape = (testing_cases, num, channels, length), "
                             "but got %d dim(s)" % len(data.shape))

        if len(labels.shape) != 3:
            raise ValueError("excepted labels.shape = (testing_cases, num, class_num), "
                             "but got %d dim(s)" % len(labels.shape))

        cases, num, channels, length = data.shape

        if few_num:
            if not isinstance(few_num, int):
                raise TypeError("expect few_num is an integer")
            if few_num < 0 or few_num > num:
                raise ValueError("expected 0 < few-num < total sampling number (%d), but got %d" % (num, few_num))
            logging.info("according to few-shot sample number, extracting few-shot set")
            perm = np.arange(num)
            np.random.shuffle(perm)
            data, labels = data[:, perm[:few_num]], labels[:, perm[:few_num], :]
            num = few_num

        if chs is not None:
            if not isinstance(chs, list) and not isinstance(chs, tuple):
                chs = [chs]
            if len(chs) > channels:
                raise ValueError("expected channels < the channels of data (%d), but got %d" % (channels, len(chs)))
            chs = [int(item) for item in chs]
            chs = np.array(chs, dtype=np.int)
            data = data[:, :, chs]
            channels = len(chs)

        wcs, cases = 2, cases // 2
        data, labels = data.reshape(cases, wcs, num, channels, length), labels.reshape(cases, wcs, num, -1)
        data, labels = data.transpose(1, 0, 2, 3, 4), labels.transpose(1, 0, 2, 3)
        data, labels = data.reshape(wcs, cases * num, channels, length), labels.reshape(wcs, cases * num, -1)
        logging.info("bearing working condition transferring set prepared, with [few-shot-num: %s] and [channels: %s]"
                     % (str(few_num), str(chs)))

        return data, labels

    def gear_working_condition_transferring(self, few_num=None, chs=None):

        if not isinstance(self.data, np.ndarray) or self.status != "gear":
            logging.info("loading testing-case-level gear dataset into memory")
            self.dataset_read_SU(data_need="gear")
            logging.info("testing-case-level gear dataset loaded !")

        data, labels = self.data, self.labels

        if len(data.shape) != 4:
            raise ValueError("excepted data.shape = (testing_cases, num, channels, length), "
                             "but got %d dim(s)" % len(data.shape))

        if len(labels.shape) != 3:
            raise ValueError("excepted labels.shape = (testing_cases, num, class_num), "
                             "but got %d dim(s)" % len(labels.shape))

        cases, num, channels, length = data.shape
        if few_num:
            if not isinstance(few_num, int):
                raise TypeError("expect few_num is an integer")

            if few_num < 0 or few_num > num:
                raise ValueError("expected 0 < few-num < total sampling number (%d), but got %d" % (num, few_num))

            logging.info("according to few-shot sample number, extracting few-shot set")

            perm = np.arange(num)
            np.random.shuffle(perm)
            data, labels = data[:, perm[:few_num]], labels[:, perm[:few_num]]
            num = few_num

        if chs is not None:
            if not isinstance(chs, list) and not isinstance(chs, tuple):
                chs = [chs]

            if len(chs) > channels:
                raise ValueError("expected channels < the channels of data (%d), but got %d" % (channels, len(chs)))

            chs = [int(item) for item in chs]
            chs = np.array(chs, dtype=np.int)
            data = data[:, :, chs]
            channels = len(chs)

        wcs, cases = 2, cases // 2
        data, labels = data.reshape(cases, wcs, num, channels, length), labels.reshape(cases, wcs, num, -1)
        data, labels = data.transpose(1, 0, 2, 3, 4), labels.transpose(1, 0, 2, 3)
        data, labels = data.reshape(wcs, cases * num, channels, length), labels.reshape(wcs, cases * num, -1)
        logging.info("gear working condition transferring set prepared, with [few-shot-num: %s] and [channels: %s]"
                     % (str(few_num), str(chs)))

        return data, labels


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s--%(name)s--%(module)s--%(levelname)s]: %(message)s")

    " fix me "
    path_project = "E:\\mechanical_fault_diagnosis_dataset\\"
    sample_num, sample_len = 300, 6000

    sudata = SoutheastBearingGear(sample_num, sample_len, path_project=path_project)
    data_bearing, labels_bearing = sudata.bearing_working_condition_transferring(few_num=60, chs=(1, 2, 3))
    data_gear, labels_gear = sudata.gear_working_condition_transferring(few_num=30, chs=3)


