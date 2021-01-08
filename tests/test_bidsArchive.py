import json
import logging
import os
from pathlib import Path
import re
import tempfile

from bids.layout.writing import build_path as bids_build_path
import nibabel as nib
import numpy as np
import pytest

from rtCommon.bidsArchive import BidsArchive
from rtCommon.bidsIncremental import BidsIncremental
from rtCommon.bidsCommon import (
    BIDS_FILE_PATH_PATTERN,
    BidsFileExtension,
    getNiftiData,
    isNiftiPath,
)

from rtCommon.errors import StateError, ValidationError

logger = logging.getLogger(__name__)

""" -----BEGIN HELPERS----- """


# Helper for checking data after append
def appendDataMatches(archive: BidsArchive, reference: BidsIncremental,
                      startIndex: int = 0, endIndex: int = -1):
    imagePath = bids_build_path(reference.imgMetadata, BIDS_FILE_PATH_PATTERN) \
        + BidsFileExtension.IMAGE.value
    imageFromArchive = archive.getImage(imagePath)

    fullImageData = getNiftiData(imageFromArchive)
    if endIndex == -1:
        endIndex = len(fullImageData)
    appendedData = fullImageData[..., startIndex:endIndex]

    appendedImage = nib.Nifti1Image(appendedData,
                                    imageFromArchive.affine,
                                    imageFromArchive.header)

    return BidsIncremental(appendedImage, reference.imgMetadata) == reference


def archiveHasMetadata(archive: BidsArchive, metadata: dict) -> bool:
    """
    Test if archive's metadata matches provided metadata dict
    """

    # Compare metadata reported by PyBids to metadata we expect has been written
    bidsLayout = archive.dataset.data
    archiveMetadata = {}
    for f in bidsLayout.get(return_type='filename'):
        if not isNiftiPath(f):
            continue
        archiveMetadata.update(
            bidsLayout.get_metadata(f, include_entities=True))
    for key, value in archiveMetadata.items():
        niftiValue = metadata.get(key, None)
        if niftiValue is None or niftiValue == value:
            continue
        # special case BIDS interpretation of int as int vs. dict has string
        elif type(value) is int and int(niftiValue) == value:
            continue
        # special case when metadata has been converted to BIDS values (seconds)
        # by BIDS-I construction
        elif int(niftiValue) / 1000 == value:
            continue
        else:
            logger.debug(f"{niftiValue}, type {type(niftiValue)} != {value}, "
                         f"type {type(value)}")
            return False

    return True

def incrementAcquisitionTime(incremental: BidsIncremental) -> None:
    """
    Increment the acquisition time in an image metadata dictionary to prepare
    for append an incremental to an archive built with the same source image.
    """
    previousAcquisitionTime = incremental.getMetadataField("AcquisitionTime")
    if previousAcquisitionTime is None:
        return
    else:
        previousAcquisitionTime = float(previousAcquisitionTime)

    trTime = incremental.getMetadataField("RepetitionTime")
    trTime = 1.0 if trTime is None else float(trTime)

    incremental.setMetadataField("AcquisitionTime",
                                previousAcquisitionTime + trTime)


""" -----BEGIN TESTS----- """


# Test archive's string output is correct
def testStringOutput(bidsArchive3D):
    logger.debug(str(bidsArchive3D))
    outPattern = r"^BIDS Layout: \S+ \| Subjects: \d+ \| Sessions: \d+ " \
                 r"\| Runs: \d+$"
    assert re.fullmatch(outPattern, str(bidsArchive3D)) is not None


# Test creating archive without a path
def testEmptyArchiveCreation(tmpdir):
    datasetRoot = Path(tmpdir, "bids-archive")
    assert BidsArchive(datasetRoot) is not None


# Test finding an image in an archive
def testFindImage(bidsArchive3D, sample3DNifti1, imageMetadata):
    imagePath = bids_build_path(imageMetadata, BIDS_FILE_PATH_PATTERN) \
        + BidsFileExtension.IMAGE.value
    archiveImage = bidsArchive3D.getImage(imagePath)
    assert archiveImage.header == sample3DNifti1.header
    assert np.array_equal(getNiftiData(archiveImage),
                          getNiftiData(sample3DNifti1))


# Test failing to find an image in an archive
def testFailFindImage(bidsArchive3D, sample3DNifti1, imageMetadata, tmpdir):
    imageMetadata['subject'] = 'nonValidSubject'
    imagePath = bids_build_path(imageMetadata, BIDS_FILE_PATH_PATTERN) \
        + BidsFileExtension.IMAGE.value
    assert bidsArchive3D.getImage(imagePath) is None

    # Test failing when dataset is empty
    datasetRoot = Path(tmpdir, "bids-archive")
    emptyArchive = BidsArchive(datasetRoot)

    with pytest.raises(StateError):
        emptyArchive.pathExists("will fail anyway")
        emptyArchive.getFilesForPath("will fail anyway")
        emptyArchive.getImage("will fail anyway")
        emptyArchive.addImage(None, "will fall anyway")
        emptyArchive.getMetadata("will fall anyway")
        emptyArchive.addMetadata({"will": "fail"}, "will fall anyway")
        emptyArchive.stripIncremental(subject="will fall anyway",
                                      session="will fall anyway",
                                      task="will fall anyway",
                                      suffix="will fall anyway",
                                      dataType="will fall anyway")


# Test images are correctly appended to an empty archive
def testEmptyArchiveAppend(validBidsI, imageMetadata, tmpdir):
    # Create in root with no BIDS-I, then append to make non-empty archive
    datasetRoot = Path(tmpdir, testEmptyArchiveAppend.__name__)
    archive = BidsArchive(datasetRoot)
    archive.appendIncremental(validBidsI)

    assert not archive.isEmpty()
    assert archiveHasMetadata(archive, imageMetadata)

    assert appendDataMatches(archive, validBidsI)


# Test images are correctly appended to an archive with just a 3-D image in it
def test3DAppend(bidsArchive3D, validBidsI, imageMetadata):
    incrementAcquisitionTime(validBidsI)
    bidsArchive3D.appendIncremental(validBidsI)
    assert archiveHasMetadata(bidsArchive3D, imageMetadata)
    assert appendDataMatches(bidsArchive3D, validBidsI, startIndex=1)


# Test appending raises error if no already existing image to append to and
# specified not to create path
def testAppendNoMakePath(bidsArchive3D, validBidsI, tmpdir):
    # Appending to empty archive
    datasetRoot = Path(tmpdir, testEmptyArchiveAppend.__name__)
    with pytest.raises(StateError):
        BidsArchive(datasetRoot).appendIncremental(validBidsI, makePath=False)

    # Appending to populated archive
    validBidsI.setMetadataField('subject', 'invalidSubject')
    validBidsI.setMetadataField('run', 42)

    with pytest.raises(ValidationError):
        bidsArchive3D.appendIncremental(validBidsI, makePath=False)


# Test appending raises error when NIfTI headers incompatible with existing
def testConflictingNiftiHeaderAppend(bidsArchive3D, sample3DNifti1, imageMetadata):
    # Modify NIfTI header in critical way (change the datatype)
    sample3DNifti1.header['datatype'] = 32  # 32=complex, should be uint16=512
    with pytest.raises(ValidationError):
        bidsArchive3D.appendIncremental(BidsIncremental(sample3DNifti1,
                                                        imageMetadata))


# Test appending raises error when image metadata incompatible with existing
def testConflictingMetadataAppend(bidsArchive3D, sample3DNifti1, imageMetadata):
    # Modify metadata in critical way (change the subject)
    imageMetadata['ProtocolName'] = 'not the same'
    with pytest.raises(ValidationError):
        """
        logging.info("Archive all files: %s",
                     json.dumps(bidsArchive3D.dataset.data.get_files(),
                                sort_keys=True, indent=4, default=str))
       """
        bidsArchive3D.appendIncremental(BidsIncremental(sample3DNifti1,
                                                        imageMetadata))
    pass


# Test images are correctly appended to an archive with a single 4-D image in it
def test4DAppend(bidsArchive4D, validBidsI, imageMetadata):
    incrementAcquisitionTime(validBidsI)
    bidsArchive4D.appendIncremental(validBidsI)

    assert archiveHasMetadata(bidsArchive4D, imageMetadata)
    assert appendDataMatches(bidsArchive4D, validBidsI, startIndex=2)


# Test images are correctly appended to an archive with a 4-D sequence in it
def testSequenceAppend(bidsArchive4D, validBidsI, imageMetadata):
    NUM_APPENDS = 2
    BIDSI_LENGTH = 2

    for i in range(NUM_APPENDS):
        incrementAcquisitionTime(validBidsI)
        bidsArchive4D.appendIncremental(validBidsI)

    imagePath = bids_build_path(imageMetadata, BIDS_FILE_PATH_PATTERN) + '.nii'
    image = bidsArchive4D.getImage(imagePath)

    shape = image.header.get_data_shape()
    assert len(shape) == 4 and shape[3] == (BIDSI_LENGTH * (1 + NUM_APPENDS))

    assert archiveHasMetadata(bidsArchive4D, imageMetadata)
    assert appendDataMatches(bidsArchive4D, validBidsI,
                             startIndex=2, endIndex=4)


# Test appending a new subject (and thus creating a new directory) to a
# non-empty BIDS Archive
def testAppendNewSubject(bidsArchive4D, validBidsI):
    preSubjects = bidsArchive4D.subjects()

    validBidsI.setMetadataField("subject", "02")
    bidsArchive4D.appendIncremental(validBidsI)

    assert len(bidsArchive4D.subjects()) == len(preSubjects) + 1

    assert appendDataMatches(bidsArchive4D, validBidsI)


# Test stripping an image off from a BIDS archive works as expected
def testStripImage(bidsArchive4D, sample3DNifti1, sampleNifti1, imageMetadata):
    incremental = bidsArchive4D.stripIncremental(
                    imageMetadata["subject"],
                    imageMetadata["session"],
                    imageMetadata["task"],
                    imageMetadata["suffix"],
                    "func")

    assert len(incremental.imageDimensions) == 4
    assert incremental.imageDimensions[-1] == 1

    reference = BidsIncremental(sample3DNifti1, imageMetadata)
    assert incremental == reference