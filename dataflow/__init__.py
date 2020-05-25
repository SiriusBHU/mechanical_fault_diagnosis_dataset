"""
    the store form of each dataset is organized as follows:
        (testing_case, working_conditions, sample_num, channels, sample_len)
        totally there has 5 dimensions

    we have 4 original database, including:
        CWRU: 10 classes
              10 testing cases corresponding to 10 classes, respectively,
              each case is tested under 4 working conditions,
              vibration signals have only 1 channel

        MFPT: 3 classes
              3 testing cases corresponding to 3 classes, respectively,
              2 of 3 cases (inner, outer fault) are tested under 6 working conditions
              1 of 3 case (normal) are tested under 3 working conditions
              vibration signals have only 1 channel

        SU_bearing: 5 classes
                    5 testing cases corresponding to 3 classes, respectively,
                    each case is tested under 2 working conditions
                    vibration signals have 8 channels, but only the 3th, 4th, 5th are valid
        SU_gear: 5 classes
                 5 testing cases corresponding to 3 classes, respectively,
                 each case is tested under 2 working conditions
                 vibration signals have 8 channels, but only the 3th, 4th, 5th are valid

        PU_data: 3 classes
                 32 testing cases, where 6 of them are normal,
                                         11 are inner fault (6 are artificial, 5 are real)
                                         12 are outer fault (7 are artificial, 5 are real)
                                         3 are combination fault (3 are real)
                 each case is tested under 4 working conditions for 20 times
                 vibration signals have only 1 channel

    after data_flow process, 4 datasets are expected to be obtained:

        CWRU: datased.shape = (10 cases, 4 conditions, num, 1 channel, len)
        MFPT: datased.shape = ( 3 cases, 6 conditions, num, 1 channel, len)  [normal data are reused]
        SU_B: datased.shape = ( 5 cases, 2 conditions, num, 8 channel, len)
        SU_G: datased.shape = ( 5 cases, 3 conditions, num, 8 channel, len)
          PU_dataset: datased.shape = (32 cases, 4 conditions, num, 1 channel, len)


"""
