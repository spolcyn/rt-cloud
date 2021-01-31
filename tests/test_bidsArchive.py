import logging
from operator import eq as opeq
import os
from pathlib import Path
import re

from bids.exceptions import (
    NoMatchError,
)
from bids.layout.writing import build_path as bids_build_path
import nibabel as nib
import numpy as np
import pytest

from rtCommon.bidsArchive import BidsArchive
from rtCommon.bidsIncremental import BidsIncremental
from rtCommon.bidsCommon import (
    BIDS_FILE_PATH_PATTERN,
    BidsFileExtension,
    filterEntities,
    getNiftiData,
    isNiftiPath,
    symmetricDictDifference,
)

from rtCommon.errors import MissingMetadataError, StateError, ValidationError

logger = logging.getLogger(__name__)

""" -----BEGIN HELPERS----- """


# Helper for checking data after append
def appendDataMatches(archive: BidsArchive, reference: BidsIncremental,
                      startIndex: int = 0, endIndex: int = -1):
    entities = filterEntities(reference.imgMetadata)
    images = archive.getImages(**entities)
    assert len(images) == 1
    imageFromArchive = images[0].get_image()

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
    bidsLayout = archive.data
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


def incrementAcquisitionValues(incremental: BidsIncremental) -> None:
    """
    Increment the acquisition values in an image metadata dictionary to prepare
    for append an incremental to an archive built with the same source image.
    """
    trTime = incremental.getMetadataField("RepetitionTime")
    trTime = 1.0 if trTime is None else float(trTime)

    fieldToIncrement = {'AcquisitionTime': trTime, 'AcquisitionNumber': 1.0}

    for field, increment in fieldToIncrement.items():
        previousValue = incremental.getMetadataField(field)
        if previousValue is None:
            continue
        else:
            previousValue = float(previousValue)
            incremental.setMetadataField(field, previousValue + increment)


""" ----- BEGIN TEST ARCHIVE QUERYING ----- """


# Test using attributes forwarded to the BIDSLayout
def testAttributeForward(bidsArchive4D):
    assert bidsArchive4D.getSubject() == bidsArchive4D.getSubjects() == ['01']
    assert bidsArchive4D.getRun() == bidsArchive4D.getRuns() == [1]
    assert bidsArchive4D.getSession() == bidsArchive4D.getSessions() == ['01']
    assert bidsArchive4D.getCeagent() == bidsArchive4D.getCeagents() == []
    assert bidsArchive4D.getDirection() == bidsArchive4D.getDirections() == []


# Test archive's string output is correct
def testStringOutput(bidsArchive3D):
    logger.debug(str(bidsArchive3D))
    outPattern = r"^Root: \S+ \| Subjects: \d+ \| Sessions: \d+ " \
                 r"\| Runs: \d+$"
    assert re.fullmatch(outPattern, str(bidsArchive3D)) is not None


# Test creating archive without a path
def testEmptyArchiveCreation(tmpdir):
    datasetRoot = Path(tmpdir, "bids-archive")
    assert BidsArchive(datasetRoot) is not None


# Test empty determiniation
def testIsEmpty(tmpdir, bidsArchive3D):
    datasetRoot = Path(tmpdir, "bids-archive")
    archive = BidsArchive(datasetRoot)
    assert archive is not None
    assert archive.isEmpty()

    assert not bidsArchive3D.isEmpty()


# Test finding an image in an archive
def testGetImages(bidsArchive3D, sample3DNifti1, bidsArchiveMultipleRuns,
                  imageMetadata):
    entities = ['subject', 'task', 'session']
    dataDict = {key: imageMetadata[key] for key in entities}

    archiveImages = bidsArchive3D.getImages(**dataDict, matchExact=False)
    assert len(archiveImages) == 1

    archiveImage = archiveImages[0].get_image()
    assert archiveImage.header == sample3DNifti1.header
    assert np.array_equal(getNiftiData(archiveImage),
                          getNiftiData(sample3DNifti1))

    # Exact match requires set of provided entities and set of entities in a
    # filename to be exactly the same (1-1 mapping); since 'run' isn't provided,
    # an exact match will fail for the multiple runs archive, which has files
    # with the 'run' entity, but will succeed for a non-exact matching, as the
    # provided entities match a subset of the file entities
    archiveImages = bidsArchiveMultipleRuns.getImages(**dataDict,
                                                      matchExact=True)
    assert archiveImages == []

    matchingDict = dataDict.copy()
    matchingDict.update({'datatype': 'func', 'suffix': 'bold', 'run': 1})
    archiveImages = bidsArchiveMultipleRuns.getImages(**matchingDict,
                                                      matchExact=True)
    assert archiveImages != []

    archiveImages = bidsArchiveMultipleRuns.getImages(**dataDict,
                                                      matchExact=False)
    assert archiveImages != []
    assert len(archiveImages) == 2


# Test failing to find an image in an archive
def testFailFindImage(bidsArchive3D, sample3DNifti1, imageMetadata, caplog):
    dataDict = {'subject': 'nonValidSubject'}
    assert bidsArchive3D.getImages(**dataDict) == []
    assert f'No images have all provided entities: {dataDict}' in caplog.text

    dataDict['subject'] = imageMetadata['subject']
    dataDict['task'] = 'invalidTask'
    assert bidsArchive3D.getImages(**dataDict) == []
    assert f'No images have all provided entities: {dataDict}' in caplog.text


# Test failing when dataset is empty
def testFailEmpty(tmpdir):
    datasetRoot = Path(tmpdir, "bids-archive")
    emptyArchive = BidsArchive(datasetRoot)

    with pytest.raises(StateError):
        emptyArchive.pathExists("will fail anyway")
        emptyArchive.getImages("will fail anyway")
        emptyArchive.addImage(None, "will fall anyway")
        emptyArchive.getMetadata("will fall anyway")
        emptyArchive.addMetadata({"will": "fail"}, "will fall anyway")
        emptyArchive.getIncremental(subject="will fall anyway",
                                    session="will fall anyway",
                                    task="will fall anyway",
                                    suffix="will fall anyway",
                                    datatype="will fall anyway")


# Test getting metadata from the archive
def testGetMetadata(bidsArchive3D, imageMetadata):
    # all entities in imageMetadata should be returned
    EXTENSION = '.nii'
    returnedMeta = bidsArchive3D.getMetadata(
        bids_build_path(imageMetadata, BIDS_FILE_PATH_PATTERN) + EXTENSION)
    imageMetadata['extension'] = EXTENSION
    imageMetadata['datatype'] = 'func'

    diff = symmetricDictDifference(returnedMeta, imageMetadata, opeq)
    assert diff == {}

# Test getting an event file from the archive
def testGetEvents(validBidsI, imageMetadata, tmpdir):
    archive = BidsArchive(tmpdir)
    archive.appendIncremental(validBidsI)

    # Get the events from the archive as a pandas data frame
    events = archive.getEvents()[0].get_df()
    assert events is not None

    # Check the required columns are present in the events file data frame
    for column in ['onset', 'duration', 'response_time']:
        assert column in events.columns


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
    assert "Nifti headers not append compatible due to mismatch in dimensions "\
           "and pixdim fields." in caplog.text
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
    other4D.header['dim'][0] = 2
    assert not BidsArchive._imagesAppendCompatible(other4D,
                                                   sample4DNifti1)


# Test metdata fields are correctly compared for append compatibility
def testMetadataValidation(imageMetadata, caplog):
    metadataCopy = imageMetadata.copy()

    # Exact copies are not compatible as some metadata must be different
    assert not BidsArchive._metadataAppendCompatible(imageMetadata,
                                                     metadataCopy)

    # Any metadata that must be different and is the same results in a failure
    differentFields = ['AcquisitionTime', 'AcquisitionNumber']
    for field in differentFields:
        oldValue = metadataCopy[field]
        metadataCopy[field] = float(oldValue) + 1
        assert not BidsArchive._metadataAppendCompatible(imageMetadata,
                                                         metadataCopy)
        metadataCopy[field] = oldValue
        assert f"Metadata matches (shouldn't) on field: {field}" in caplog.text

    # Modify fields and ensure append now possible
    for field in differentFields:
        metadataCopy[field] = float(metadataCopy[field]) + 1
    assert BidsArchive._metadataAppendCompatible(imageMetadata,
                                                 metadataCopy)
    # Test failure on sample of fields that must be the same
    matchFields = ["Modality", "MagneticFieldStrength", "ImagingFrequency",
                   "Manufacturer", "ManufacturersModelName", "InstitutionName",
                   "InstitutionAddress", "DeviceSerialNumber", "StationName",
                   "BodyPartExamined", "PatientPosition", "EchoTime",
                   "ProcedureStepDescription", "SoftwareVersions",
                   "MRAcquisitionType", "SeriesDescription", "ProtocolName",
                   "ScanningSequence", "SequenceVariant", "ScanOptions",
                   "SequenceName", "SpacingBetweenSlices", "SliceThickness",
                   "ImageType", "RepetitionTime", "PhaseEncodingDirection",
                   "FlipAngle", "InPlanePhaseEncodingDirectionDICOM",
                   "ImageOrientationPatientDICOM", "PartialFourier"]

    for field in matchFields:
        oldValue = metadataCopy.get(field, None)

        # If field not present, append should work
        if oldValue is None:
            assert BidsArchive._metadataAppendCompatible(imageMetadata,
                                                         metadataCopy)
        # If field is present, modify and ensure failure
        else:
            metadataCopy[field] = "not a valid value by any stretch of the word"
            assert metadataCopy[field] != oldValue
            assert not BidsArchive._metadataAppendCompatible(imageMetadata,
                                                             metadataCopy)
            metadataCopy[field] = oldValue
            assert f"Metadata doesn't match on field: {field}" in caplog.text

    # Test append-compatible when only one side has a particular metadata value
    for field in (matchFields + differentFields):
        for metadataDict in [imageMetadata, metadataCopy]:
            oldValue = metadataDict.pop(field, None)
            if oldValue is None:
                continue
            assert BidsArchive._metadataAppendCompatible(imageMetadata,
                                                         metadataCopy)
            metadataDict[field] = oldValue


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
    incrementAcquisitionValues(validBidsI)
    bidsArchive3D.appendIncremental(validBidsI)
    assert archiveHasMetadata(bidsArchive3D, imageMetadata)
    assert appendDataMatches(bidsArchive3D, validBidsI, startIndex=1)


# Test appending raises error if no already existing image to append to and
# specified not to create path
def testAppendNoMakePath(bidsArchive3D, validBidsI, tmpdir):
    # Append to empty archive specifying not to make any files or directories
    datasetRoot = Path(tmpdir, testEmptyArchiveAppend.__name__)
    with pytest.raises(StateError):
        BidsArchive(datasetRoot).appendIncremental(validBidsI, makePath=False)

    # Append to populated archive in a way that would require new directories
    # and files without allowing it
    validBidsI.setMetadataField('subject', 'invalidSubject')
    validBidsI.setMetadataField('run', 42)

    with pytest.raises(StateError):
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
    incrementAcquisitionValues(validBidsI)
    bidsArchive4D.appendIncremental(validBidsI)

    assert archiveHasMetadata(bidsArchive4D, imageMetadata)
    assert appendDataMatches(bidsArchive4D, validBidsI, startIndex=2)


# Test images are correctly appended to an archive with a 4-D sequence in it
def testSequenceAppend(bidsArchive4D, validBidsI, imageMetadata):
    NUM_APPENDS = 2
    BIDSI_LENGTH = 2

    for i in range(NUM_APPENDS):
        incrementAcquisitionValues(validBidsI)
        bidsArchive4D.appendIncremental(validBidsI)

    image = bidsArchive4D.getImages(
        matchExact=False, **filterEntities(imageMetadata))[0].get_image()

    shape = image.header.get_data_shape()
    assert len(shape) == 4 and shape[3] == (BIDSI_LENGTH * (1 + NUM_APPENDS))

    assert archiveHasMetadata(bidsArchive4D, imageMetadata)
    assert appendDataMatches(bidsArchive4D, validBidsI,
                             startIndex=2, endIndex=4)


# Test appending a new subject (and thus creating a new directory) to a
# non-empty BIDS Archive
def testAppendNewSubject(bidsArchive4D, validBidsI):
    preSubjects = bidsArchive4D.getSubjects()

    validBidsI.setMetadataField("subject", "02")
    bidsArchive4D.appendIncremental(validBidsI)

    assert len(bidsArchive4D.getSubjects()) == len(preSubjects) + 1

    assert appendDataMatches(bidsArchive4D, validBidsI)


""" ----- BEGIN TEST IMAGE STRIPPING ----- """


# Test stripping an image off a BIDS archive works as expected
def testStripImage(bidsArchive3D, bidsArchive4D, sample3DNifti1, sample4DNifti1,
                   imageMetadata):
    # 3D Case
    reference = BidsIncremental(sample3DNifti1, imageMetadata)
    incremental = bidsArchive3D.getIncremental(
        imageMetadata["subject"],
        imageMetadata["task"],
        imageMetadata["suffix"],
        "func",
        session=imageMetadata["session"])

    # 3D image still results in 4D incremental
    assert len(incremental.imageDimensions) == 4
    assert incremental.imageDimensions[3] == 1

    assert incremental == reference

    # 4D Case
    # Both the first and second image in the 4D archive should be identical
    reference = BidsIncremental(sample3DNifti1, imageMetadata)
    for index in range(0, 2):
        incremental = bidsArchive4D.getIncremental(
                        imageMetadata["subject"],
                        imageMetadata["task"],
                        imageMetadata["suffix"],
                        "func",
                        sliceIndex=index,
                        session=imageMetadata["session"])

        assert len(incremental.imageDimensions) == 4
        assert incremental.imageDimensions[3] == 1

        assert incremental == reference


# Test stripping image from BIDS archive fails when no matching images are
# present in the archive
def testStripNoMatchingImage(bidsArchive4D, imageMetadata):
    imageMetadata['subject'] = 'notPresent'
    with pytest.raises(NoMatchError):
        incremental = bidsArchive4D.getIncremental(
            imageMetadata["subject"],
            imageMetadata["task"],
            imageMetadata["suffix"],
            "func",
            session=imageMetadata["session"])

        assert incremental is None


# Test stripping image from BIDS archive raises warning when no matching
# metadata is present in the archive
def testStripNoMatchingMetadata(bidsArchive4D, imageMetadata, caplog, tmpdir):
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

    # Without the sidecar metadata, not enough information for an incremental
    errorText = r"Archive lacks required metadata for BIDS Incremental " \
                r"creation: .*"
    with pytest.raises(MissingMetadataError, match=errorText):
        bidsArchive4D.getIncremental(imageMetadata["subject"],
                                     imageMetadata["task"],
                                     imageMetadata["suffix"],
                                     "func",
                                     session=imageMetadata["session"])


# Test strip with an out-of-bounds slice index for the matching image (could be
# either non-0 for 3D or beyond bounds for a 4D)
def testStripSliceIndexOutOfBounds(bidsArchive3D, bidsArchive4D, imageMetadata,
                                   caplog):
    # Negative case
    outOfBoundsIndex = -1
    incremental = bidsArchive3D.getIncremental(
        imageMetadata["subject"],
        imageMetadata["task"],
        imageMetadata["suffix"],
        "func",
        sliceIndex=-1,
        session=imageMetadata["session"])

    assert f"Slice index must be >= 0 (got {outOfBoundsIndex})" in caplog.text
    assert incremental is None

    # 3D case
    outOfBoundsIndex = 1
    incremental = bidsArchive3D.getIncremental(
        imageMetadata["subject"],
        imageMetadata["task"],
        imageMetadata["suffix"],
        "func",
        sliceIndex=outOfBoundsIndex,
        session=imageMetadata["session"])

    assert f"Matching image was a 3-D NIfTI; time index {outOfBoundsIndex} " \
           f"too high for a 3-D NIfTI (must be 0)" in caplog.text
    assert incremental is None

    # 4D case
    outOfBoundsIndex = 4
    archiveLength = 2
    incremental = bidsArchive4D.getIncremental(
        imageMetadata["subject"],
        imageMetadata["task"],
        imageMetadata["suffix"],
        "func",
        sliceIndex=outOfBoundsIndex,
        session=imageMetadata["session"])

    assert f"Image index {outOfBoundsIndex} too large for NIfTI volume of " \
           f"length {archiveLength}" in caplog.text
    assert incremental is None


# Test stripping when files are found, but none match provided parameters
# exactly
def testStripNoParameterMatch(bidsArchive4D, imageMetadata, caplog):
    # Test entity values that don't exist in the archive
    errorText = r"Unable to find any data in archive that matches" \
                r" all provided entities \(got: \{.*?\}\)"
    with pytest.raises(NoMatchError, match=errorText):
        incremental = bidsArchive4D.getIncremental(
            imageMetadata["subject"],
            imageMetadata["task"],
            imageMetadata["suffix"],
            "func",
            session=imageMetadata['session'],
            run=2)

        assert incremental is None

    # Test non-existent task, subject, session, and suffix in turn
    modificationPairs = {'subject': 'nonExistentSubject',
                         'session': 'nonExistentSession',
                         'task': 'nonExistentSession',
                         'suffix': 'notBoldCBvOrPhase'}

    for argName, argValue in modificationPairs.items():
        oldValue = imageMetadata[argName]
        imageMetadata[argName] = argValue

        with pytest.raises(NoMatchError, match=errorText):
            incremental = bidsArchive4D.getIncremental(
                subject=imageMetadata["subject"],
                task=imageMetadata["task"],
                suffix=imageMetadata["suffix"],
                datatype="func",
                session=imageMetadata['session'])

            assert incremental is None

        imageMetadata[argName] = oldValue
