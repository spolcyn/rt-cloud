import logging
import os
from pathlib import Path
import re

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


""" ----- BEGIN TEST ARCHIVE QUERYING ----- """


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


""" ----- BEGIN TEST APPENDING ----- """


# Test NIfTI headers are correctly compared for append compatibility
def testNiftiHeaderValidation(sample4DNifti1, sample3DNifti1, sample2DNifti1,
                              caplog):
    # Prepare test infrastructure
    original3DHeader = sample3DNifti1.header.copy()
    original4DHeader = sample4DNifti1.header.copy()

    other3D = nib.Nifti1Image(sample3DNifti1.dataobj,
                              sample3DNifti1.affine,
                              sample3DNifti1.header)
    assert other3D.header == original3DHeader

    other4D = nib.Nifti1Image(sample4DNifti1.dataobj,
                              sample4DNifti1.affine,
                              sample4DNifti1.header)
    assert other4D.header == original4DHeader

    """ Test field values """
    # Test equal headers
    assert BidsArchive._imagesAppendCompatible(sample4DNifti1, other4D)

    # Test unequal headers on variety of fields that must match
    fieldsToModify = ["intent_code", "dim_info", "scl_slope", "sform_code"]

    for field in fieldsToModify:
        fieldArray = other4D.header[field]
        oldValue = fieldArray.copy()

        if np.sum(np.isnan(fieldArray)) > 0:
            fieldArray = np.zeros(1)
        else:
            fieldArray = fieldArray + 1
        other4D.header[field] = fieldArray

        assert not BidsArchive._imagesAppendCompatible(sample4DNifti1, other4D)
        assert "Nifti headers don't match on field: " + field in caplog.text

        other4D.header[field] = oldValue

    """ Test special cases for dimensions and pixel dimensions being non-equal
    but still append compatible """
    # First three dimensions and pixel dimensions equal
    assert BidsArchive._imagesAppendCompatible(sample3DNifti1, sample4DNifti1)

    # Dimension 4 of the 3D image should not matter
    for i in range(0, 100):
        sample3DNifti1.header["dim"][4] = i
        assert BidsArchive._imagesAppendCompatible(sample3DNifti1,
                                                   sample4DNifti1)

    sample3DNifti1.header["dim"] = np.copy(original3DHeader["dim"])
    assert sample3DNifti1.header == original3DHeader

    """ Test special cases for dimensions and pixel dimensions being non-equal
    and not append compatible """
    # Ensure all headers are in their original states
    assert sample4DNifti1.header == original4DHeader
    assert other4D.header == original4DHeader
    assert sample3DNifti1.header == original3DHeader
    assert other3D.header == original3DHeader

    # 4D with non-matching first 3 dimensions should fail
    other4D.header["dim"][1:4] = other4D.header["dim"][1:4] * 2
    assert not BidsArchive._imagesAppendCompatible(sample4DNifti1,
                                                   other4D)
    # Reset
    other4D.header["dim"][1:4] = original4DHeader["dim"][1:4]
    assert other4D.header == original4DHeader

    # 3D and 4D in which first 3 dimensions don't match
    other3D.header["dim"][1:3] = other3D.header["dim"][1:3] * 2
    assert not BidsArchive._imagesAppendCompatible(sample4DNifti1,
                                                   other3D)

    # Reset
    other3D.header["dim"][1:3] = original3DHeader["dim"][1:3]
    assert other3D.header == original3DHeader

    # 2D and 4D are one too many dimensions apart
    assert not BidsArchive._imagesAppendCompatible(sample2DNifti1,
                                                   sample4DNifti1)


# Test metdata fields are correctly compared for append compatibility
def testMetadataValidation():
    pytest.skip()


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
def testConflictingNiftiHeaderAppend(bidsArchive3D, sample3DNifti1,
                                     imageMetadata):
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


""" ----- BEGIN TEST IMAGE STRIPPING ----- """


# Test stripping an image off a BIDS archive works as expected
def testStripImage(bidsArchive3D, bidsArchive4D, sample3DNifti1, sample4DNifti1,
                   imageMetadata):
    # 3D Case
    reference = BidsIncremental(sample3DNifti1, imageMetadata)
    incremental = bidsArchive3D.stripIncremental(
        imageMetadata["subject"],
        imageMetadata["session"],
        imageMetadata["task"],
        imageMetadata["suffix"],
        "func")

    # 3D image still results in 4D incremental
    assert len(incremental.imageDimensions) == 4
    assert incremental.imageDimensions[3] == 1

    assert incremental == reference

    # 4D Case
    # Both the first and second image in the 4D archive should be identical
    reference = BidsIncremental(sample3DNifti1, imageMetadata)
    for index in range(0, 2):
        incremental = bidsArchive4D.stripIncremental(
                        imageMetadata["subject"],
                        imageMetadata["session"],
                        imageMetadata["task"],
                        imageMetadata["suffix"],
                        "func",
                        sliceIndex=index)

        assert len(incremental.imageDimensions) == 4
        assert incremental.imageDimensions[3] == 1

        assert incremental == reference


# Test stripping image from BIDS archive fails when no matching images are
# present in the archive
def testStripNoMatchingImage(bidsArchive4D, imageMetadata):
    imageMetadata['subject'] = 'notPresent'
    incremental = bidsArchive4D.stripIncremental(
        imageMetadata["subject"],
        imageMetadata["session"],
        imageMetadata["task"],
        imageMetadata["suffix"],
        "func")

    assert incremental is None


# Test stripping image from BIDS archive raises warning when no matching
# metadata is present in the archive
def testStripNoMatchingMetdata(bidsArchive4D, imageMetadata, caplog, tmpdir):
    # Create path to sidecar metadata file
    relPath = bids_build_path(imageMetadata, BIDS_FILE_PATH_PATTERN) + \
        BidsFileExtension.METADATA.value

    absPath = None
    files = os.listdir(tmpdir)
    for fname in files:
        if "dataset" in fname:
            absPath = Path(tmpdir, fname, relPath)
            break

    # Remove the sidecar metadata file
    os.remove(absPath)
    bidsArchive4D._update()

    # Configure logging to capture the warning
    caplog.set_level(logging.WARNING, logger="rtCommon.bidsArchive")

    # Without the sidecar metadata, not enough information for an incremental
    with pytest.raises(ValidationError):
        bidsArchive4D.stripIncremental(imageMetadata["subject"],
                                       imageMetadata["session"],
                                       imageMetadata["task"],
                                       imageMetadata["suffix"],
                                       "func")

    # Check the logging for the warning message
    assert "Archive didn't contain any matching metadata" in caplog.text


# Test strip with an out-of-bounds slice index for the matching image (could be
# either non-0 for 3D or beyond bounds for a 4D)
def testStripSliceIndexOutOfBounds(bidsArchive3D, bidsArchive4D, imageMetadata,
                                   caplog):
    # Negative case
    outOfBoundsIndex = -1
    incremental = bidsArchive3D.stripIncremental(
        imageMetadata["subject"],
        imageMetadata["session"],
        imageMetadata["task"],
        imageMetadata["suffix"],
        "func",
        sliceIndex=-1)

    assert f"Slice index must be >= 0 (got {outOfBoundsIndex})" in caplog.text
    assert incremental is None

    # 3D case
    outOfBoundsIndex = 1
    incremental = bidsArchive3D.stripIncremental(
        imageMetadata["subject"],
        imageMetadata["session"],
        imageMetadata["task"],
        imageMetadata["suffix"],
        "func",
        sliceIndex=outOfBoundsIndex)

    assert f"Matching image was a 3-D NIfTI; time index {outOfBoundsIndex} " \
           f"too high for a 3-D NIfTI (must be 0)" in caplog.text
    assert incremental is None

    # 4D case
    outOfBoundsIndex = 4
    archiveLength = 2
    incremental = bidsArchive4D.stripIncremental(
        imageMetadata["subject"],
        imageMetadata["session"],
        imageMetadata["task"],
        imageMetadata["suffix"],
        "func",
        sliceIndex=outOfBoundsIndex)

    assert f"Image index {outOfBoundsIndex} too large for NIfTI volume of " \
           f"length {archiveLength}" in caplog.text
    assert incremental is None


# Test stripping when files are found, but none match provided parameters
# exactly
def testStripNoParameterMatch(bidsArchive4D, imageMetadata, caplog):
    # Test non-existent otherLabels
    incremental = bidsArchive4D.stripIncremental(
        imageMetadata["subject"],
        imageMetadata["session"],
        imageMetadata["task"],
        imageMetadata["suffix"],
        "func",
        otherLabels={'run': 2})

    assert incremental is None
    assert "Failed to find matching image in BIDS Archive " \
        "for provided metadata" in caplog.text

    # Test non-existent task, subject, session, and suffix in turn
    modificationPairs = {'subject': 'nonExistentSubject',
                         'session': 'nonExistentSession',
                         'task': 'nonExistentSession',
                         'suffix': 'notBoldCBvOrPhase'}

    for argName, argValue in modificationPairs.items():
        oldValue = imageMetadata[argName]
        imageMetadata[argName] = argValue

        incremental = bidsArchive4D.stripIncremental(
            subject=imageMetadata["subject"],
            session=imageMetadata["session"],
            task=imageMetadata["task"],
            suffix=imageMetadata["suffix"],
            dataType="func")

        assert incremental is None
        assert "Failed to find matching image in BIDS Archive " \
            "for provided metadata" in caplog.text

        imageMetadata[argName] = oldValue
