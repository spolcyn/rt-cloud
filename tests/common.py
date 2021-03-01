# common.py
# Constants and functions shared by tests

import os

""" Imaging inputs """
test_inputDirName = 'test_input'
test_dicomFile = '001_000013_000005.dcm'
test_dicomTruncFile = 'trunc_001_000013_000005.dcm'
test_nifti1_3DFile = 'test_input_3d_func_ses-01_task-story_run-01_bold.nii'
test_nifti2_3DFile = 'test_input_3d_nifti2_func_ses-01_task-story_run-01_bold.nii'
test_nifti1_4DFile = 'test_input_4d_func_ses-01_task-story_run-01_bold.nii'
test_nifti2_4DFile = 'test_input_4d_nifti2_func_ses-01_task-story_run-01_bold.nii'

# absolute paths derived from above names
testRootPath = os.path.dirname(__file__)
rootPath = os.path.dirname(testRootPath)
test_inputDirPath = os.path.join(testRootPath, test_inputDirName)
test_dicomPath = os.path.join(test_inputDirPath, test_dicomFile)
test_dicomTruncPath = os.path.join(test_inputDirPath, test_dicomTruncFile)
test_sampleProjectPath = os.path.join(rootPath, 'projects/sample')
test_sampleProjectDicomPath = os.path.join(test_sampleProjectPath, 
    "dicomDir/20190219.0219191_faceMatching.0219191_faceMatching/")
test_3DNifti1Path = os.path.join(test_inputDirPath, test_nifti1_3DFile)
test_3DNifti2Path = os.path.join(test_inputDirPath, test_nifti2_3DFile)
test_4DNifti1Path = os.path.join(test_inputDirPath, test_nifti1_4DFile)
test_4DNifti2Path = os.path.join(test_inputDirPath, test_nifti2_4DFile)
