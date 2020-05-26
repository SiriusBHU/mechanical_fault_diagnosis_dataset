"""
    Paderborn University Bearing DataSet ReadFile
    Author: Sirius HU
    Created Date: 2019.10.05

    Data Introduction:
        here we have 32 testing bearings,
            among them, 6 are normal ones
                        12 with artificial damage (made manually)
                        14 with real damage (caused during operating lifetime)

            the 6 bearings are selected after going through different lifetime and working conditions

            the 12 bearings are artificially damaged by EDM (电火花加工), Drilling (钻孔), EE (电刻蚀)
            among them, 5 are inner race defects
                        7 are outer race defects

            the 14 bearings are damaged during accelerated lifetime operation,
            most of them suffer from fatigue(pitting), and few are plastic deformation(indentations)
            among them, 6 are inner race defects
                        5 are outer race defects
                        3 are inner-outer mixed defects

        after selection, all the 32 bearings are operated under 4 different working conditions for 20 times
        so here we have 32 directories, each has 80 .mat files recording the electricity and vibration info.
        besides, each dir. also have two introduction files

        for more details, refer to:
            https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter/
            Lessmeier C, Kimotho J K, Zimmer D, et al., Condition monitoring of bearing damage in electromechanical
            drive systems by using motor current signals of electric motors: a benchmark data set for data-driven
            classification: Proceedings of the European conference of the prognostics and health management society,
            2016[C].

    Note:
        considering some bearing has both inner and outer fault, the corresponding (one-hot) label
        such as (0, 1, 0, 0) is soft as (0, 0.8, 0.2, 0) [or (0, 0.2. 0.8, 0)] according to the faulty size

"""

import logging
import os
import scipy.io as sio
import numpy as np
import dataflow.util as ut


class PaderbornBearing(object):

    """
        using case:
            pudata = PaderbornBearing(sample_num, sample_len, path_project)
            # as soon as we declare the class PaderbornBearing,
            # it will check if there has a prepared testing-case-level dataset
            # if not, the dataset will be generated during __init__

            data_wc, labels_wc = pudata.working_condition_transferring(few_num=60)
            # after __init__, we can generate transferring dataset, by two funcs:
            #       1. working_condition_transferring
            #       2. artificial_real_transferring
            # they will first check if the dataset has been loaded in the memory,
            # then check if we need to generate few-shot set (few_num=None or not),
            # and then form transferring set

            here label 1 0 0 means health
                 label 0 1 0 means inner fault
                 label 0 0 1 means outer fault
    """

    # set dataset url
    __url = "http://groups.uni-paderborn.de/kat/BearingDataCenter/"

    def __init__(self,
                 sample_num=None, sample_len=None,
                 path_project=None,
                 path_mat=None, path_npy=None, path_cache=None):
        """
            check store path of the Paderborn bearing data
            check the whether the 32 testing-case-level datasets exist
            if not, generate them
        """

        self.path_mat = path_mat
        self.path_npz = path_npy
        self.path_cache = path_cache
        self.path = path_project

        if not self.path:
            self.path = os.getcwd()

        if not self.path_mat:
            self.path_mat = self.path + "\\dataset\\Diagnostics\\PU_data\\"
        if not self.path_cache:
            self.path_cache = self.path + "\\datacache\\PU_dataset\\"
        if not self.path_npz:
            self.path_npz = self.path + "\\datacache\\PU_dataset\\_origin\\"

        # check dataset related directories  do or not exist
        if not os.path.exists(self.path_mat):
            logging.warning("no original .mat path")
            os.makedirs(self.path_mat)

        if not os.path.exists(self.path_cache):
            logging.warning("no formulated dataset path")
            os.makedirs(self.path_cache)

        if not os.path.exists(self.path_npz):
            logging.warning("no original .npz path")
            os.makedirs(self.path_npz)

        # check the dataset has or hasn't been prepared
        # if not, generate testing-case-level dataset
        if len(os.listdir(self.path_cache)) == 1:
            logging.warning("no formulated dataset")

            if not len(os.listdir(self.path_npz)):
                logging.warning("no original .npz files")

                if not len(os.listdir(self.path_mat)):
                    raise FileExistsError("no original .mat files\n"
                                          "please download the files from url: %s\n"
                                          "and unzip it in the dir %s"
                                          % (self.__url, self.path_mat))

                logging.info("convert orignal .mat files to original .npz files")
                self.origin_mat2npz_PU()
                logging.info(".npz files has been generated !")

            if not sample_num or not sample_len:
                raise KeyError("keys [sample_num] and [sample_len] are needed for sampling")

            logging.info("sampling orignal .npz files to formulate raw dataset with testing cases level dataset")
            self.dataset_prepare_PU(sample_num, sample_len)

        logging.info("testing-case-level dataset has been prepared !")

        # prepare a interface to load testing-case-level dataset
        self.data, self.labels = None, None

    def origin_mat2npz_PU(self):

        """
            here we aim to extract the vibration signals in .mat files,
            and restore them in .npy files for each bearing (total 32),
            each one constructed .npy file contains 80 records of 4-working-condition (each 20) testing signals
        """

        # check the dir contain all the data (32 bearing)
        sub_dirs = os.listdir(self.path_mat)
        if len(sub_dirs) != 32:
            raise ValueError("the dir number not match the testing bearing number: %d != 32" % len(sub_dirs))

        # for each bearing, we extract all the vibration info into a .npy file
        # the file contain an array with shape (80, 2e5) sampling points
        # each 20 sub-array are generated from one working condition
        for i, sub_dir in enumerate(sub_dirs):

            bearing_files = os.listdir(self.path_mat + "\\" + sub_dir)
            os.chdir(self.path_mat + "\\" + sub_dir)
            sub_data = []
            min_len = 1e10

            for j, mat_name in enumerate(bearing_files[2:]):

                a = sio.loadmat(mat_name)
                a = a[mat_name[:-4]]
                a = a["Y"]
                a = a[0][0][0][6][2]
                min_len = np.shape(a)[1] if np.shape(a)[1] < min_len else min_len
                sub_data.append(a)
            sub_data = [item[0, :min_len] for item in sub_data]
            sub_data = np.array(sub_data)

            os.chdir(self.path_npz)
            np.savez(sub_dir + ".npz", arr=sub_data)
        os.chdir(self.path)

        return 0

    def filename_PU(self):

        files = os.listdir(self.path_npz)
        labels = [1, 1, 1, 1, 1, 1,
                  2, 2, 2, 2, 2, 2, 2,
                  0, 0, 0, 0, 0, 0,
                  1, 1, 1, 1, 1,
                  1, 1, 2,
                  2, 2, 2, 2, 2]

        return files, labels

    def dataset_prepare_PU(self, sample_num, sample_len):

        """
            here 32 testing-case-level sub-sets will be generated and grepped into one dataset
            this dataset will be prepared for two kind of transferring scenarios, including:
                working condition transferring
                artificial-to-real transferring
        """

        logging.info("generating new dataset with each case has (sample_num = %d, sample_len = %d)"
                     % (sample_num, sample_len))

        # get filenames and the labels corresponding to each file
        filenames, filelabels = self.filename_PU()

        # read each file, sampling examples from original vibration signals
        os.chdir(self.path_npz)
        sample_num = int(sample_num/20)
        data, labels = [], []
        for k, file in enumerate(filenames):

            signals = np.load(file)["arr"]
            if len(signals.shape) != 2:
                raise ValueError("excepted signals have 2 dims, but got %d dim(s)" % len(signals.shape))

            if np.shape(signals)[0] != 80:
                raise ValueError("excepted signals' number is 80 , but got %d" % np.shape(signals)[0])

            num, length = signals.shape
            signals = signals.reshape(num, 1, length)
            samples = []

            for i in range(num):
                samples.append(ut.signal_split_as_samples(signals[i], sample_len, sample_num))

            data.append(np.array(samples).reshape(4, 20 * sample_num, 1, sample_len))
            sample_labels = np.ones((80 * sample_num, 1)) * filelabels[k]
            sample_labels = ut.onehot_encoding(sample_labels, 3)
            sample_labels = sample_labels.reshape(4, 20 * sample_num, -1)

            # considering the combination of inner and outer fault
            if "Inner_Outer" in file:
                sample_labels[:, :, 1] = 0.8
                sample_labels[:, :, 2] = 0.2
            if "Outer_Inner" in file:
                sample_labels[:, :, 1] = 0.2
                sample_labels[:, :, 2] = 0.8

            labels.append(sample_labels)

        data, labels = np.array(data), np.array(labels)
        os.chdir(self.path_cache)
        np.savez("PU_1D_signal.npz", data=data, labels=labels)
        os.chdir(self.path)

        logging.info("new dataset has been stored")

        return 0

    def dataset_read_PU(self):

        dataset = np.load(self.path_cache + "\\PU_1D_signal.npz")
        self.data = dataset["data"]
        self.labels = dataset["labels"]

    def working_condition_transferring(self, few_num=None):

        # 这里需要先确保 testing-case-level 的 32 个数据集先加载到内存中，
        # 后续操作都不会动内存中的数据，而只改变指向内存的地址（或者说改变引用方式）
        if not isinstance(self.data, np.ndarray):
            logging.info("loading testing-case-level dataset into memory")
            self.dataset_read_PU()
            logging.info("testing-case-level dataset loaded !")

        # 这里一定要新建变量 data，labels，
        # 其相当于将 self.data 和 self.labels 中指针的指向关系 copy 到 data 和 labels 两个变量中
        # 改变 data, labels 中指针的指向关系，并不会影响 self.data 和 self.labels 中指针的指向关系
        # 但改变 data, labels 指向的数据，则 self.data, self.labels 指向的数据也将改变
        data, labels = self.data, self.labels

        if len(data.shape) != 5:
            raise ValueError("excepted data.shape = (testing_cases, working_conditions, num, channels, length), "
                             "but got %d dim(s)" % len(data.shape))

        if len(labels.shape) != 4:
            raise ValueError("excepted labels.shape = (testing_cases, working_conditions, num, class_num), "
                             "but got %d dim(s)" % len(labels.shape))

        cases, wcs, num, channels, length = data.shape
        if few_num:
            if few_num < 0 or few_num > num:
                raise ValueError("expected 0 < few-num < total sampling number (%d), but got %d" % (num, few_num))
            logging.info("according to few-shot sample number, extracting few-shot set")
            perm = np.arange(num)
            np.random.shuffle(perm)
            data, labels = data[:, :, perm[:few_num]], labels[:, :, perm[:few_num], :]
            num = few_num

        data, labels = data.transpose(1, 0, 2, 3, 4), labels.transpose(1, 0, 2, 3)
        data, labels = data.reshape(wcs, cases * num, channels, length), labels.reshape(wcs, cases * num, -1)
        logging.info("working condition transferring set prepared, with [few-shot-num: %s]" % str(few_num))

        return data, labels

    def artificial_real_transferring(self, few_num=None):

        # 这里需要先确保 testing-case-level 的 32 个数据集先加载到内存中，
        # 后续操作都不会动内存中的数据，而只改变指向内存的地址（或者说改变引用方式）
        if not isinstance(self.data, np.ndarray):
            logging.info("loading testing-case-level dataset into memory")
            self.dataset_read_PU()
            logging.info("testing-case-level dataset loaded !")

        # 这里一定要新建变量 data，labels，
        # 其相当于将 self.data 和 self.labels 中指针的指向关系 copy 到 data 和 labels 两个变量中
        # 改变 data, labels 中指针的指向关系，并不会影响 self.data 和 self.labels 中指针的指向关系
        # 但改变 data, labels 指向的数据，则 self.data, self.labels 指向的数据也将改变
        data, labels = self.data, self.labels

        if len(data.shape) != 5:
            raise ValueError("excepted data.shape = (testing_cases, working_conditions, num, channels, length), "
                             "but got %d dim(s)" % len(data.shape))

        if len(labels.shape) != 4:
            raise ValueError("excepted labels.shape = (testing_cases, working_conditions, num, class_num), "
                             "but got %d dim(s)" % len(labels.shape))

        cases, wcs, num, channels, length = data.shape
        if few_num:
            if few_num < 0 or few_num > num:
                raise ValueError("expected 0 < few-num < total sampling number (%d), but got %d" % (num, few_num))
            logging.info("according to few-shot sample number, extracting few-shot set")
            perm = np.arange(num)
            np.random.shuffle(perm)
            data, labels = data[:, :, perm[:few_num]], labels[:, :, perm[:few_num], :]
            num = few_num

        arti_range = (12, 14, 16,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11)
        real_range = (13, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
        data_arti, labels_arti = np.array([data[i] for i in arti_range]), np.array([labels[i] for i in arti_range])
        data_real, labels_real = np.array([data[i] for i in real_range]), np.array([labels[i] for i in real_range])

        data_arti = data_arti.reshape(15 * wcs * num, channels, length)
        data_real = data_real.reshape(17 * wcs * num, channels, length)
        labels_arti = labels_arti.reshape(15 * wcs * num, -1)
        labels_real = labels_real.reshape(17 * wcs * num, -1)

        data, labels = [data_arti, data_real], [labels_arti, labels_real]
        logging.info("artificial-to-real transferring set prepared, with [few-shot-num: %s]" % str(few_num))

        return data, labels


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s--%(name)s--%(module)s--%(levelname)s]: %(message)s")

    " fix me "
    path_project = "E:\\mechanical_fault_diagnosis_dataset\\"
    sample_num, sample_len = 300, 6000

    # here label 1 0 0 means health
    #      label 0 1 0 means inner fault
    #      label 0 0 1 means outer fault
    pudata = PaderbornBearing(sample_num, sample_len, path_project=path_project)
    data_wc, labels_wc = pudata.working_condition_transferring(60)
    data_ar, labels_ar = pudata.artificial_real_transferring(30)

    # os.chdir(path_npy)
    # a = np.load("Artificial_Outer_KA05.npz")
    # a = a["arr"]

