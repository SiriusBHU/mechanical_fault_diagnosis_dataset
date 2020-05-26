

import zipfile
import os
path_txt = "E:\\mechanical_fault_diagnosis_dataset\\dataset\\Diagnostics\\CWRU_data"

_file = "CWRU.zip"
with zipfile.ZipFile(os.path.join(path_txt, _file), 'r') as _unzip_ref:
    _unzip_ref.extractall(path_txt)



os.remove(_file)