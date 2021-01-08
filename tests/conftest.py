# Note: Could modularize this further by creating a fixtures dir and importing
# See: https://gist.github.com/peterhurford/09f7dcda0ab04b95c026c60fa49c2a68
import json
import logging
import os
from pathlib import Path
from random import randint

from bids.layout.writing import build_path as bids_build_path
import nibabel as nib
import pydicom
import pytest

from common import (
    test_dicomPath,
    test_3DNifti1Path,
    test_3DNifti2Path,
    test_4DNifti1Path,
    test_4DNifti2Path
)
from rtCommon.imageHandling import (
    readDicomFromFile,
    readNifti
)
from rtCommon.bidsArchive import BidsArchive
from rtCommon.bidsCommon import (
    DEFAULT_DATASET_DESC,
    BIDS_DIR_PATH_PATTERN,
    BIDS_FILE_PATTERN,
    BIDS_FILE_PATH_PATTERN,
    getNiftiData
)
from rtCommon.bidsIncremental import BidsIncremental
import rtCommon.bidsLibrary as bl

logger = logging.getLogger(__name__)


""" BEGIN DICOM RELATED FIXTURES """


# Dictionary of some fields of the read-in DICOM image
@pytest.fixture
def dicomMetadataSample() -> dict:
    sample = {}
    sample["ContentDate"] = "20190219"
    sample["ContentTime"] = "124758.653000"
    sample["RepetitionTime"] = 1500
    sample["StudyDescription"] = "Norman_Mennen^5516_greenEyes"
    sample["StudyInstanceUID"] = \
        "1.3.12.2.1107.5.2.19.45031.30000019021215313940500000046"

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
    newData = getNiftiData(nifti)
    # max positive value of 2 byte, signed short used in Nifti header for
    # storing dimension information
    newData = newData.flatten()[:2**15 - 1]
    return nib.Nifti1Image(newData, nifti.affine)


# 3-D NIfTI 1 image derived from the test DICOM image
@pytest.fixture(scope='function')
def sample3DNifti1():
    return readNifti(test_3DNifti1Path)


# 3-D NIfTI 2 image derived from the test DICOM image
@pytest.fixture(scope='function')
def sample3DNifti2():
    return readNifti(test_3DNifti2Path)


# 4-D NIfTI 1 image derived from concatting the test DICOM image with itself
@pytest.fixture(scope='function')
def sample4DNifti1():
    return readNifti(test_4DNifti1Path)


# 4-D NIfTI 2 image derived from concatting the test DICOM image with itself
@pytest.fixture(scope='function')
def sampleNifti2():
    return readNifti(test_4DNifti2Path)


@pytest.fixture
def imageMetadata(dicomImageMetadata):
    """
    Dictionary with all required metadata to construct a BIDS-Incremental, as
    well as extra metadata extracted from the test DICOM image.
    """
    meta = {'subject': '01', 'task': 'faces', 'suffix': 'bold',  # REQUIRED
            'session': '01', 'run': '1'}  # EXTRACTED
    meta.update(dicomImageMetadata)  # DICOM
    return meta


@pytest.fixture(scope='function')
def validBidsI(sample4DNifti1, imageMetadata):
    """
    Constructs and returns a known-valid BIDS-Incremental using known metadata.
    """
    return BidsIncremental(image=sample4DNifti1,
                           imageMetadata=imageMetadata)


def archiveWithImage(image, metadata: dict, tmpdir):
    """
    Create an archive on disk by hand with the provided image and metadata
    """

    # Create ensured empty directory
    while True:
        id = str(randint(0, 1e6))
        rootPath = Path(tmpdir, f"dataset-{id}/")
        if not Path.exists(rootPath):
            rootPath.mkdir()
            break

    # Create the archive by hand, with default readme and dataset description
    Path(rootPath, 'README').write_text("README for pytest")
    Path(rootPath, 'dataset_description.json') \
        .write_text(json.dumps(DEFAULT_DATASET_DESC))

    # Write the nifti image & metadata
    dataPath = Path(rootPath, bids_build_path(metadata, BIDS_DIR_PATH_PATTERN))
    dataPath.mkdir(parents=True)

    filenamePrefix = bids_build_path(metadata, BIDS_FILE_PATTERN)
    imagePath = Path(dataPath, filenamePrefix + '.nii')
    metadataPath = Path(dataPath, filenamePrefix + '.json')

    nib.save(image, imagePath)
    metadataPath.write_text(json.dumps(metadata))

    # Create an archive from the directory and return it
    return BidsArchive(rootPath)


# BIDS Archive with a 3-D image in it
@pytest.fixture(scope='function')
def bidsArchive3D(tmpdir, sample3DNifti1, imageMetadata):
    adjustTimeUnits(imageMetadata)
    return archiveWithImage(sample3DNifti1, imageMetadata, tmpdir)


# BIDS Archive with a 4-D image in it
@pytest.fixture(scope='function')
def bidsArchive4D(tmpdir, sample4DNifti1, imageMetadata):
    return archiveWithImage(sample4DNifti1, imageMetadata, tmpdir)


""" END BIDS RELATED FIXTURES """
