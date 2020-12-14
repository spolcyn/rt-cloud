# Note: Could modularize this further by creating a fixtures dir and importing
# See: https://gist.github.com/peterhurford/09f7dcda0ab04b95c026c60fa49c2a68
import os
import tempfile

import pydicom
import pytest

from common import test_dicomFilePath
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

@pytest.yield_fixture
def sampleNiftiImage():
    dicomPath = os.path.join(os.path.dirname(__file__), test_dicomFilePath)
    niftiPath = os.path.join(tempfile.gettempdir(), "test.nii")
    convertDicomFileToNifti(dicomPath, niftiPath)

    # 'Yield' will return the Nifti image now, and delay the running of
    # subsequent code (i.e., the removal of the Nifti image on disk) until the
    # end-of-test tear-down is being done
    # This ensures the test doesn't create clutter, but is able to access the
    # necessary data during the test
    yield readNifti(niftiPath)

    os.remove(niftiPath)

@pytest.fixture
def imageMetadataDict():
    """
    Dictionary with all required metadata to construct a BIDS-Incremental, as
    well as extra metadata extracted from the test DICOM image.
    """
    return {'subject': '01', 'task': 'story', 'suffix': 'bold',  # REQUIRED
            'session': '01', 'run': '01'}  # EXTRA

@pytest.fixture
def validBidsI(sampleNiftiImage, imageMetadataDict, dicomImageMetadata):
    """
    Constructs and returns a known-valid BIDS-Incremental using known metadata.
    """
    return BidsIncremental(image=sampleNiftiImage,
                           subject=imageMetadataDict["subject"],
                           task=imageMetadataDict["task"],
                           suffix=imageMetadataDict["suffix"],
                           imgMetadata=dicomImageMetadata)

""" END BIDS RELATED FIXTURES """
