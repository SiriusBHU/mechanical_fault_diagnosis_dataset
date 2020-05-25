"""
    Author: Sirius HU
    Created Date: 2019.11.10

    Introduction:
        this is a using case to tell how to load data from each dataset.
        first you should copy the 'dataflow' directory to your porject
        and then you can use the funcs to load data stored in a fixed place regardless of the location of your project.
"""

from dataflow.Archi_Energy_data_load import *
from dataflow.CWRU_data_load import *
from dataflow.MFPT_data_load import *
from dataflow.PU_data_load import *
from dataflow.SU_data_load import *
import logging


# prepare logger
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s--%(name)s--%(module)s--%(levelname)s]: %(message)s",)
                    #filename="log_1.txt")
# set the location of this project
# it can be a fixed place in your PC, so every time you set up a new project,
# you can use the same data from this fixed place
path_project = "E:\\mechanical_fault_diagnosis_dataset\\"

# Architecture Energy data
ae_data = ArchitectureEnergyData(archi_batch_size=10, time_len=1440, total_num=3000, path_project=path_project)
archi_train, weather_train, energy_train = ae_data.K_shot_dataset_read(choice="train", few_num=500)
archi_test, weather_test, energy_test = ae_data.K_shot_dataset_read(choice="test", few_num=500)


# CWRU bearing data
sample_num, sample_len = 500, 2048
cwdata = CaseWesternBearing(sample_num, sample_len, path_project=path_project)
cwdata.dataset_prepare_CWRU(sample_num, sample_len)
data_wc, labels_wc = cwdata.working_condition_transferring()
data_ar, labels_ar = cwdata.fault_extent_transferring()


# MFPT bearing data
sample_num, sample_len = 500, 2048
mfptdata = MFPTBearing(sample_num, sample_len, path_project=path_project)
mfptdata.dataset_prepare_MFPT(sample_num, sample_len)
data, labels = mfptdata.working_condition_transferring()


# PU bearing data
sample_num, sample_len = 300, 6000
pudata = PaderbornBearing(sample_num, sample_len, path_project=path_project)
data_wc, labels_wc = pudata.working_condition_transferring(60)
data_ar, labels_ar = pudata.artificial_real_transferring(30)


# SU bearing & gear data
sample_num, sample_len = 300, 6000
sudata = SoutheastBearingGear(sample_num, sample_len, path_project=path_project)
data_bearing, labels_bearing = sudata.bearing_working_condition_transferring(few_num=60, chs=(1, 2, 3))
data_gear, labels_gear = sudata.gear_working_condition_transferring(few_num=30, chs=3)

