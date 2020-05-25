"""
    Author: Sirius HU
    Created Date: 2019.11.06

    Architecture Energy DataSet Preparation
"""

import pandas as pd
import numpy as np
import logging
import os


class ArchitectureEnergyData(object):

    """
        using case:
            ae_data = ArchitectureEnergyData(path_npz=path_npz,
                                             path_cache=path_cache,
                                             path_origin_data=path_origin_data,
                                             archi_batch_num=10, time_len=1440, total_num=3000)
            # as soon as we declare the class  ArchitectureEnergyData,
            # it will check if there has a prepared dataset
            # if not, it will generate it during __init__


            ae_data.dataset_prepare_AE(archi_batch_size=20, time_len=720, total_num=5000)
            # if you do not satisfied with your current dataset, you can use func
                    dataset_prepare_AE(archi_batch_size, time_len, total_num)
            # to re-generate a new dataset with new (archi_batch_size, time_len, total_num)


            archi_train, weather_train, energy_train = ae_data.K_shot_dataset_read(choice="train", few_num=500)
            archi_test, weather_test, energy_test = ae_data.K_shot_dataset_read(choice="test", few_num=500)
            # after __init__, we can generate a K_shot dataset, by func:
            #       K_shot_dataset_read
            # it will first check if the dataset has been loaded in the memory,
            # then check if we need to generate a K_shot set (few_num=None or an integer),
            # and then form a dataset with K numbers
    """
    def __init__(self, logger_level=None,
                 archi_batch_size=None, time_len=None, total_num=None,
                 path_project=None,
                 path_npz=None,
                 path_cache=None,
                 path_origin_data=None):

        self.path_npz = path_npz
        self.path_cache = path_cache
        self.path_origin_data = path_origin_data
        self.path = path_project

        self.logger = logging.getLogger(__name__)
        if not logger_level:
            logger_level = logging.INFO
        self.logger.setLevel(logger_level)

        if not self.path:
            self.path = os.getcwd()

        if not self.path_origin_data:
            self.path_origin_data = self.path + "\\dataset\\Architecture_Energy\\"
        self.path_weather = self.path_origin_data + "\\param_weather.xlsx"
        self.path_archi = self.path_origin_data + "\\param_archi.xlsx"
        self.dir_energy = self.path_origin_data + "\\data_archi_energy"

        if not self.path_cache:
            self.path_cache = self.path + "\\datacache\\Archi_Energy\\"

        if not self.path_npz:
            self.path_npz = self.path + "\\datacache\\Archi_Energy\\_origin"

        # check dataset related directories  do or not exist
        if not os.path.exists(self.path_origin_data):
            logging.warning("no original energy data path")
            os.makedirs(self.path_origin_data)

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

                if not len(os.listdir(self.path_origin_data)):
                    raise FileExistsError("no original energy data")

                logging.info("extract original .xlsx files to original .npz files")
                self.origin_mat2npz_AE()
                logging.info(".npz files has been generated !")

            if not archi_batch_size or not time_len or not total_num:
                raise KeyError("keys [archi_batch_num], [time_len] and [total_num] are needed for sampling")

            logging.info("sampling original .npz files to formulate raw data-set")
            self.dataset_prepare_AE(archi_batch_size, time_len, total_num)
        logging.info("data-set has been prepared !")

        self.training_set, self.testing_set = None, None

    def origin_mat2npz_AE(self):

        # warning, mix precision
        weather = pd.read_excel(self.path_weather).values[:, 1:]
        archi = pd.read_excel(self.path_archi, header=None).values[:, 1: 9]

        # to do this is to eliminate the inverse impact of mixing precision,
        # e.g. int and float are stored in the same array
        weather, archi = np.array(weather, dtype=np.float), np.array(archi, dtype=np.float)

        os.chdir(self.dir_energy)
        energy_data = []
        for i in range(archi.shape[0]):
            file = "aa%d.csv" % i
            energy_data.append(pd.read_csv(file).values[:, 1:])
        energy_data = np.array(energy_data, dtype=np.float).transpose(0, 2, 1)

        os.chdir(self.path_npz)
        np.savez("original_archi_energy_data.npz",
                 weather=weather,
                 archi=archi,
                 energy_data=energy_data)
        os.chdir(self.path)

        return

    def dataset_prepare_AE(self, archi_batch_size, time_len, total_num, train_rate=0.5):

        logging.info("generating new dataset with each case has (archi_batch_num = %d, time_len = %d, total_num = %d)"
                     % (archi_batch_size, time_len, total_num))
        os.chdir(self.path_npz)

        # note that, if the stored data is mixing precision, allow_pickle must be set as True
        # (warning!! it will increase the read-time-consumption a lot !!)
        data = np.load("original_archi_energy_data.npz", allow_pickle=False)
        weather, archi, energy_data = data["weather"], data["archi"], data["energy_data"]

        archi_num = archi.shape[0]
        perm = np.arange(archi_num)
        np.random.shuffle(perm)
        train_indexes, test_indexes = perm[: int(archi_num * train_rate)], perm[int(archi_num * train_rate):]
        train_num, test_num = int(total_num * train_rate), total_num - int(total_num * train_rate)

        training_set = {"archi": [], "weather": [], "energy": []}
        testing_set = {"archi": [], "weather": [], "energy": []}

        self._data_split(training_set,
                         archi_batch_size, time_len,
                         train_num, train_indexes,
                         weather, archi, energy_data)

        self._data_split(testing_set,
                         archi_batch_size, time_len,
                         test_num, test_indexes,
                         weather, archi, energy_data)

        os.chdir(self.path_cache)
        np.savez("Archi_Energy_TrainingSet.npz",
                 archi=training_set['archi'], weather=training_set["weather"], energy=training_set['energy'])
        np.savez("Archi_Energy_TestingSet.npz",
                 archi=testing_set['archi'], weather=testing_set["weather"], energy=testing_set['energy'])

        logging.info("new dataset has been stored")

        return 0

    def _data_split(self,
                    dataset,
                    archi_batch_size, time_len, total_num,
                    total_indexes,
                    weather, archi, energy_data):

        _indexes = np.arange(total_indexes.shape[0])

        if "archi" not in dataset.keys() or \
           "weather" not in dataset.keys() or \
           "energy" not in dataset.keys():
            raise KeyError("expect dict 'dataset' has keys ('archi, 'weather', 'energy'), "
                           "but got ", dataset.keys())

        for i in range(total_num):

            np.random.shuffle(_indexes)
            rand_batch_index = total_indexes[_indexes[:archi_batch_size]]

            _sample_archi, _sample_weather, _sample_energy = archi[rand_batch_index], [], []
            for index in rand_batch_index:
                rand_seed = np.random.randint(0, weather.shape[0] - time_len)
                _sample_weather.append(weather[rand_seed: rand_seed + time_len])
                _sample_energy.append(energy_data[index, :, rand_seed: rand_seed + time_len])

            dataset["archi"].append(_sample_archi)
            dataset["weather"].append(np.array(_sample_weather))
            dataset["energy"].append(np.array(_sample_energy))

        for key in dataset.keys():
            dataset[key] = np.array(dataset[key])

        return 0

    def dataset_read_AE(self):

        training_io = np.load(self.path_cache + "\\Archi_Energy_TrainingSet.npz")
        testing_io = np.load(self.path_cache + "\\Archi_Energy_TestingSet.npz")

        self.training_set = (training_io["archi"], training_io["weather"], training_io["energy"])
        self.testing_set = (testing_io["archi"], testing_io["weather"], testing_io["energy"])

    def K_shot_dataset_read(self, choice="train", few_num=None):

        if not isinstance(self.training_set, tuple) or not isinstance(self.testing_set, tuple):
            self.dataset_read_AE()

        if choice == "train":
            archi, weather, energy = self.training_set
        elif choice == "test":
            archi, weather, energy = self.testing_set
        else:
            raise ValueError("expected choice is within ['train', 'test'], but got %s\n" % str(choice),
                             "it aims to select the dataset you want to read")

        perm = np.arange(archi.shape[0])
        if few_num:
            if few_num > archi.shape[0]:
                raise ValueError("few_num should be lower than the total num of samples")
            np.random.shuffle(perm)
            perm = perm[:few_num]

        return archi[perm], weather[perm], energy[perm]


if __name__ == "__main__":

    " fix me "
    path_project = "D:\\DataSet_Preparation\\"

    ae_data = ArchitectureEnergyData(path_project=path_project,
                                     archi_batch_size=10, time_len=1440, total_num=3000)
    archi_train, weather_train, energy_train = ae_data.K_shot_dataset_read(choice="train", few_num=500)
    archi_test, weather_test, energy_test = ae_data.K_shot_dataset_read(choice="test", few_num=500)
