# Note: Could modularize this further by creating a fixtures dir and importing
# See: https://gist.github.com/peterhurford/09f7dcda0ab04b95c026c60fa49c2a68
import json
import logging
import os
from pathlib import Path
from random import randint

import nibabel as nib
import pydicom
import pytest

from common import (
    test_dicomPath,
    test_3DNifti1Path,
    test_3DNifti2Path,
    test_nifti1Path,
    test_nifti2Path
)
from rtCommon.imageHandling import (
    readDicomFromFile,
    readNifti
)
from rtCommon.bidsArchive import BidsArchive
from rtCommon.bidsIncremental import BidsIncremental
from rtCommon.bidsCommon import DEFAULT_DATASET_DESC
import rtCommon.bidsLibrary as bl

logger = logging.getLogger(__name__)


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
                                           test_dicomPath))
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


# 2-D NIfTI 1 image corrupted from the test DICOM image
@pytest.fixture
def sample2DNifti():
    nifti = readNifti(test_3DNifti1Path)
    newData = nifti.get_fdata().flatten()
    # max positive value of 2 byte, signed short used in Nifti header for
    # storing dimension information
    newData = newData[:2**15 - 1]
    return nib.Nifti1Image(newData, nifti.affine)


# 3-D NIfTI 1 image derived from the test DICOM image
@pytest.fixture
def sample3DNifti1():
    return readNifti(test_3DNifti1Path)


# 3-D NIfTI 2 image derived from the test DICOM image
@pytest.fixture
def sample3DNifti2():
    return readNifti(test_3DNifti2Path)


# 4-D NIfTI 1 image derived from concatting the test DICOM image with itself
@pytest.fixture
def sampleNifti1():
    return readNifti(test_nifti1Path)


# 4-D NIfTI 2 image derived from concatting the test DICOM image with itself
@pytest.fixture
def sampleNifti2():
    return readNifti(test_nifti2Path)


@pytest.fixture
def imageMetadataDict(dicomImageMetadata):
    """
    Dictionary with all required metadata to construct a BIDS-Incremental, as
    well as extra metadata extracted from the test DICOM image.
    """
    meta = {'subject': '01', 'task': 'story', 'suffix': 'bold',  # REQUIRED
            'session': '01', 'run': '1'}  # EXTRACTED
    meta.update(dicomImageMetadata)  # DICOM
    return meta


@pytest.fixture
def validBidsI(sampleNifti1, imageMetadataDict):
    """
    Constructs and returns a known-valid BIDS-Incremental using known metadata.
    """
    return BidsIncremental(image=sampleNifti1,
                           imageMetadata=imageMetadataDict)


# BIDS Archeve with a 3-D image in it
@pytest.fixture
def bidsArchive3D(tmpdir, sample3DNifti1, imageMetadataDict):
    # Create ensured empty directory
    while True:
        id = str(randint(0, 1e6))
        rootPath = Path(tmpdir, f"dataset-{id}/")
        if not Path.exists(rootPath):
            rootPath.mkdir()
            break

    logger.debug("Root path: %s", rootPath)

    # Create the archive by hand, with default readme and dataset description
    Path(rootPath, 'README').write_text("README for" + bidsArchive3D.__name__)
    Path(rootPath, 'dataset_description.json') \
        .write_text(json.dumps(DEFAULT_DATASET_DESC))

    # Write the nifti 3D image & metadata
    dataPath = Path(rootPath, "sub-01", "ses-01", "func")
    logger.debug("Data path: %s", dataPath)
    dataPath.mkdir(parents=True)
    filenamePrefix = "sub-01_task-story_bold"
    nib.save(sample3DNifti1, Path(dataPath, filenamePrefix + '.nii'))
    metadataPath = Path(dataPath, filenamePrefix + '.json')
    metadataPath.write_text(json.dumps(imageMetadataDict))

    # Create an archive from the directory and return it
    return BidsArchive(rootPath)


""" END BIDS RELATED FIXTURES """
