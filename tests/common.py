# common.py
# Constants and functions shared by tests

import os

""" Imaging inputs """
test_inputDir = 'test_input'
test_dicomFile = '001_000005_000100.dcm'
test_dicomTruncFile = 'trunc_001_000005_000100.dcm'
test_nifti1_3DFile = 'test_input_3d_func_ses-01_task-story_run-01_bold.nii'
test_nifti2_3DFile = 'test_input_3d_nifti2_func_ses-01_task-story_run-01_bold.nii'
test_nifti1File = 'test_input_4d_func_ses-01_task-story_run-01_bold.nii'
test_nifti2File = 'test_input_4d_nifti2_func_ses-01_task-story_run-01_bold.nii'

# absolute paths derived from above names
test_inputDirPath = os.path.join(os.path.dirname(__file__), test_inputDir)
test_dicomPath = os.path.join(test_inputDirPath, test_dicomFile)
test_3DNifti1Path = os.path.join(test_inputDirPath, test_nifti1_3DFile)
test_3DNifti2Path = os.path.join(test_inputDirPath, test_nifti2_3DFile)
test_nifti1Path = os.path.join(test_inputDirPath, test_nifti1File)
test_nifti2Path = os.path.join(test_inputDirPath, test_nifti2File)
