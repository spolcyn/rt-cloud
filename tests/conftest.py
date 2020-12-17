# Note: Could modularize this further by creating a fixtures dir and importing
# See: https://gist.github.com/peterhurford/09f7dcda0ab04b95c026c60fa49c2a68
import os
import tempfile

import pydicom
import pytest

from common import test_dicomFilePath, test_niftiFilePath
from rtCommon.imageHandling import (
    readDicomFromFile,
    convertDicomFileToNifti,
    readNifti
)
from rtCommon.bidsIncremental import BidsIncremental
import rtCommon.bidsLibrary as bl

""" BEGIN DICOM RELATED FIXTURES """

# Dictionary of some fields of the read-in DICOM image
@pytest.fixture
def dicomMetadataSample() -> dict:
    sample = {}
    sample["ContentDate"] = "20190521"
    sample["ContentTime"] = "131107.519000"
    sample["RepetitionTime"] = 1500
    sample["StudyDescription"] = "Norman_Mennen^5516_greenEyes"
    sample["StudyInstanceUID"] = \
        "1.3.12.2.1107.5.2.19.45031.30000019051622212064900000040"

    return sample

# PyDicom image read in from test DICOM file
@pytest.fixture
def dicomImage(dicomMetadataSample) -> pydicom.dataset.Dataset:
    dicom = readDicomFromFile(os.path.join(os.path.dirname(__file__),
                                           test_dicomFilePath))
    assert dicom is not None

    # Test a sampling of fields to ensure proper read
    for field, value in dicomMetadataSample.items():
        assert getattr(dicom, field) == value

    return dicom

# Public metadata for test DICOM file
@pytest.fixture
def dicomImageMetadata(dicomImage):
    public, _ = bl.getMetadata(dicomImage)
    return public

""" END DICOM RELATED FIXTURES """

""" BEGIN BIDS RELATED FIXTURES """

# 4-D NIfTI image derived from concatting the test DICOM image with itself
@pytest.fixture
def sampleNiftiImage():
    return readNifti(test_niftiFilePath)

@pytest.fixture
def imageMetadataDict(dicomImageMetadata):
    """
    Dictionary with all required metadata to construct a BIDS-Incremental, as
    well as extra metadata extracted from the test DICOM image.
    """
    meta = {'subject': '01', 'task': 'story', 'suffix': 'bold',  # REQUIRED
            'session': '01', 'run': '01'}  # EXTRACTED
    meta.update(dicomImageMetadata)  # DICOM
    return meta

@pytest.fixture
def validBidsI(sampleNiftiImage, imageMetadataDict):
    """
    Constructs and returns a known-valid BIDS-Incremental using known metadata.
    """
    return BidsIncremental(image=sampleNiftiImage,
                           imageMetadata=imageMetadataDict)

""" END BIDS RELATED FIXTURES """
