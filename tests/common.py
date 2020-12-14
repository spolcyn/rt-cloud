# common.py
# Constants and functions shared by tests

import os

""" Imaging inputs """
test_inputDir = 'test_input'
test_dicomFile = '001_000005_000100.dcm'
test_dicomTruncFile = 'trunc_001_000005_000100.dcm'

test_dicomFilePath = os.path.join(test_inputDir, test_dicomFile)
