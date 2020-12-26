# Creates the various test NIfTI files used in the test suite from the source
# DICOM image

import os
import sys
import logging

import nibabel as nib
import numpy as np

# Add base project path (two directories up)
currPath = os.path.dirname(os.path.realpath(__file__))
rootPath = os.path.dirname(currPath)
sys.path.append(rootPath)

from common import (
    test_dicomPath,
    test_3DNifti1Path,
    test_3DNifti2Path,
    test_nifti1Path,
    test_nifti2Path
)
from rtCommon.imageHandling import convertDicomFileToNifti

logging.basicConfig(level=logging.INFO)

"""
Delete existing existing NIfTI files with target names to avoid dcm2niix
creating lots of duplicates with different names
"""
for path in [test_3DNifti1Path, test_3DNifti2Path,
             test_nifti1Path, test_nifti2Path]:
    if os.path.exists(path):
        logging.info("Removing existing: %s", path)
        os.remove(path)

"""
Create base 3D NIfTI1 file all others are created from
"""
convertDicomFileToNifti(test_dicomPath, test_3DNifti1Path)
nifti1 = nib.load(test_3DNifti1Path)

# Extract the TR time, then eliminate pixel dimension data past 3rd dimension,
# as a 3D image really should only have 3D data, and having more can complicate
# later comparisons.

# this currently works with dcm2niix and the DICOM we have, but is not robust
TR_TIME = nifti1.header['pixdim'][4]
nifti1.header['pixdim'][4:] = 1
nib.save(nifti1, test_3DNifti1Path)

"""
Create NIfTI2 version of 3D base
"""
nifti2 = nib.Nifti2Image(nifti1.dataobj, nifti1.affine, nifti1.header)
nib.save(nifti2, test_3DNifti2Path)

"""
Create 4D Nifti1 from base 3D Nifti1
"""
nifti1_4D = nib.concat_images([nifti1, nifti1])
nifti1_4D.header["pixdim"][4] = TR_TIME
nib.save(nifti1_4D, test_nifti1Path)

"""
Create 4D Nifti2 from 3D Nifti2
"""
nifti2_4D = nib.concat_images([nifti2, nifti2])
nifti2_4D.header["pixdim"][4] = TR_TIME
nib.save(nifti2_4D, test_nifti2Path)

"""
Validate created Nifti files by comparing headers and data that should match
"""

""" Helpers for validation """
# https://brainder.org/2015/04/03/the-nifti-2-file-format/
NIFTI2_REMOVED_FIELDS = ['data_type', 'db_name', 'extents', 'session_error',
                         'regular', 'glmin', 'glmax']
NIFTI2_CHANGED_FIELDS = ['sizeof_hdr', 'magic']


def headersMatch(niftiA, niftiB,
                 ignoredKeys: list = [],
                 specialFieldToHandler: dict = {}) -> bool:
    header1 = niftiA.header
    header2 = niftiB.header

    for key in header1:
        if key in ignoredKeys:
            continue

        v1 = header1.get(key, None)
        v2 = header2.get(key, None)

        if np.array_equal(v1, v2):
            continue
        # Check for nan equality
        else:
            if np.issubdtype(v1.dtype, np.inexact) and \
                    np.allclose(v1, v2, atol=0.0, equal_nan=True):
                continue
            # If key is special and handler returns true, continue
            elif key in specialFieldToHandler and \
                    specialFieldToHandler[key](v1, v2):
                continue
            else:
                logging.info("--------------------\n"
                             "Difference found!"
                             "Key: %s\nHeader 1: %s\nHeader 2: %s", key, v1, v2)
                return False

    return True


def dataMatch(niftiA, niftiB) -> bool:
    return np.array_equal(niftiA.dataobj, niftiB.dataobj)


# Used when dimensions will increased by one in the 3D to 4D conversion
def dim3Dto4DHandler(v1: np.ndarray, v2: np.ndarray) -> bool:
    return v1[0] + 1 == v2[0] and v1[4] + 1 == v2[4]


# Used when pixdim is different in 4th dimension for a 4D image vs. 3D
def pixdim3Dto4DHandler(v1: np.ndarray, v2: np.ndarray) -> bool:
    return np.array_equal(v1[1:3], v2[1:3])


handlerMap3Dto4D = {'dim': dim3Dto4DHandler, 'pixdim': pixdim3Dto4DHandler}

""" Actual validation """

""" 3D Nifti2 """
ignoredKeys = NIFTI2_REMOVED_FIELDS + NIFTI2_CHANGED_FIELDS
errorString = "{} for Nifti1 3D and Nifti2 3D did not match"

assert type(nib.load(test_3DNifti2Path)) is nib.Nifti2Image
assert headersMatch(nifti1, nifti2, ignoredKeys=ignoredKeys), \
    errorString.format("Headers")
assert dataMatch(nifti1, nifti2), errorString.format("Image data")

""" 4D Nifti1 """
errorString = "{} for Nifti1 3D and Nifti1 4D did not match"

assert headersMatch(nifti1, nifti1_4D, specialFieldToHandler=handlerMap3Dto4D),\
    errorString.format("Headers")

""" 4D Nifti2 """
errorString = "{} for Nifti2 3D and Nifti2 4D did not match"

assert headersMatch(nifti2, nifti2_4D, specialFieldToHandler=handlerMap3Dto4D),\
    errorString.format("Headers")

""" 4D Nifti1 and 4D Nifti2 data """
errorString = "{} for Nifti1 4D and Nifti2 4D did not match"

ignoredKeys = NIFTI2_REMOVED_FIELDS + NIFTI2_CHANGED_FIELDS
assert headersMatch(nifti1_4D, nifti2_4D, ignoredKeys=ignoredKeys), \
    errorString.format("Image data")
assert dataMatch(nifti1_4D, nifti2_4D), errorString.format("Headers")
