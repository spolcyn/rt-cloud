# common.py
# Constants and functions shared by tests

import os

""" Imaging inputs """
test_inputDir = 'test_input'
test_dicomFile = '001_000005_000100.dcm'
test_dicomTruncFile = 'trunc_001_000005_000100.dcm'
test_niftiFile = 'test_input_4d_func_ses-01_task-story_run-01_bold.nii'

# absolute paths derived from above names
test_inputDirPath = os.path.join(os.path.dirname(__file__), test_inputDir)
test_dicomFilePath = os.path.join(test_inputDirPath, test_dicomFile)
test_niftiFilePath = os.path.join(test_inputDirPath, test_niftiFile)
