import logging
import os
import tempfile

import nibabel as nib
from bids.layout.writing import build_path as bids_build_path

from rtCommon.bidsArchive import BidsArchive
from rtCommon.bidsIncremental import BidsIncremental
from rtCommon.bidsCommon import (
    BIDS_FILE_PATH_PATTERN,
    BidsFileExtension,
    getNiftiData,
    isNiftiPath,
)

logger = logging.getLogger(__name__)


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


# Test creating archive without a path
def testEmptyArchiveCreation():
    datasetRoot = os.path.join(tempfile.gettempdir(), "bids-archive")
    assert BidsArchive(datasetRoot) is not None


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


# Test images are correctly appended to an empty archive
def testEmptyArchiveAppend(validBidsI, imageMetadata, tmpdir):
    # Create in root with no BIDS-I, then append to make non-empty archive
    datasetRoot = os.path.join(tmpdir, testEmptyArchiveAppend.__name__)
    archive = BidsArchive(datasetRoot)
    archive.appendIncremental(validBidsI)

    assert not archive.isEmpty()
    assert archiveHasMetadata(archive, imageMetadata)

    assert appendDataMatches(archive, validBidsI)


# Test images are correctly appended to an archive with just a 3-D image in it
def test3DAppend(bidsArchive3D, validBidsI, imageMetadata):
    bidsArchive3D.appendIncremental(validBidsI)
    assert archiveHasMetadata(bidsArchive3D, imageMetadata)
    assert appendDataMatches(bidsArchive3D, validBidsI, startIndex=1)


# Test images are correctly appended to an archive with a single 4-D image in it
def test4DAppend(bidsArchive4D, validBidsI, imageMetadata):
    bidsArchive4D.appendIncremental(validBidsI)
    assert archiveHasMetadata(bidsArchive4D, imageMetadata)
    assert appendDataMatches(bidsArchive4D, validBidsI, startIndex=2)


# Test images are correctly appended to an archive with a 4-D sequence in it
def testSequenceAppend(bidsArchive4D, validBidsI, imageMetadata):
    NUM_APPENDS = 2
    BIDSI_LENGTH = 2

    for i in range(NUM_APPENDS):
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
