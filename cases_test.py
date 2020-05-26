from dataflow.CWRU_data_load import CaseWesternBearing
from dataflow.MFPT_data_load import MFPTBearing
from dataflow.PU_data_load import PaderbornBearing
from dataflow.SU_data_load import SoutheastBearingGear
import logging


logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s--%(name)s--%(module)s--%(levelname)s]: %(message)s")
path_project = "E:\\AAA"    # "E:\\mechanical_fault_diagnosis_dataset"
sample_num, sample_len = 500, 2048

cwdata = CaseWesternBearing(sample_num, sample_len, path_project=path_project)
mfptdata = MFPTBearing(sample_num, sample_len, path_project=path_project)
# pudata = PaderbornBearing(sample_num, sample_len, path_project=path_project)
sudata = SoutheastBearingGear(sample_num, sample_len, path_project=path_project)
