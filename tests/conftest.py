# Note: Could modularize this further by creating a fixtures dir and importing
# See: https://gist.github.com/peterhurford/09f7dcda0ab04b95c026c60fa49c2a68
import os

import pydicom
import pytest

from common import test_dicomFilePath
from rtCommon.imageHandling import readDicomFromFile

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

""" END DICOM RELATED FIXTURES """

""" BEGIN BIDS RELATED FIXTURES """

""" END BIDS RELATED FIXTURES """
