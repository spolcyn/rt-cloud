from copy import deepcopy
import io
import json
import logging
import os
import pickle
import shutil
import tempfile

from bids_validator import BIDSValidator
import pytest
import nibabel as nib
import numpy as np

from rtCommon.bidsIncremental import BidsIncremental
from rtCommon.bidsCommon import BidsFileExtension as BidsFileExtension
from rtCommon.errors import ValidationError

logger = logging.getLogger(__name__)


# Test that construction fails for image metadata missing required fields
def testInvalidConstruction(sample2DNifti, sampleNifti1, imageMetadataDict):
    # Test empty image
    with pytest.raises(ValidationError):
        BidsIncremental(image=None,
                        imageMetadata=imageMetadataDict)

    # Test 2-D image
    with pytest.raises(ValidationError):
        BidsIncremental(image=sample2DNifti,
                        imageMetadata=imageMetadataDict)

    # Test incomplete metadata
    protocolName = imageMetadataDict.pop("ProtocolName")
    for key in BidsIncremental.requiredImageMetadata:
        logger.debug("Key: %s", key)
        value = imageMetadataDict.pop(key)

        assert not BidsIncremental.isCompleteImageMetadata(imageMetadataDict)
        with pytest.raises(ValidationError):
            BidsIncremental(image=sampleNifti1,
                            imageMetadata=imageMetadataDict)

        imageMetadataDict[key] = value
    imageMetadataDict["ProtocolName"] = protocolName

    # Test too-large repetition and echo times
    for key in ["RepetitionTime", "EchoTime"]:
        original = imageMetadataDict[key]
        imageMetadataDict[key] = 10**6

        with pytest.raises(ValidationError):
            BidsIncremental(image=sampleNifti1,
                            imageMetadata=imageMetadataDict)

        imageMetadataDict[key] = original

    # Test non-image object
    with pytest.raises(ValidationError):
        BidsIncremental(image="definitely not an image",
                        imageMetadata=imageMetadataDict)


# Test that valid arguments produce a BIDS incremental
def testValidConstruction(sample3DNifti, sampleNifti1, sampleNifti2, imageMetadataDict):
    # 3-D should be promoted to 4-D
    assert BidsIncremental(sample3DNifti, imageMetadataDict) is not None

    # Both Nifti1 and Nifti2 images should work
    assert BidsIncremental(sampleNifti1, imageMetadataDict) is not None
    assert BidsIncremental(sampleNifti2, imageMetadataDict) is not None

    # If the metadata provides a RepetitionTime or EchoTime that works without
    # adjustment, the construction should still work
    repetitionTimeKey= "RepetitionTime"
    original = imageMetadataDict[repetitionTimeKey]
    imageMetadataDict[repetitionTimeKey] = 1.5
    assert BidsIncremental(sampleNifti1, imageMetadataDict) is not None
    imageMetadataDict[repetitionTimeKey] = original

# Test that the string output of the BIDS-I is as expected
def testStringOutput(validBidsI):
    imageShape = str(validBidsI.imageDimensions())
    keyCount = len(validBidsI.imgMetadata.keys())
    version = validBidsI.version
    assert str(validBidsI) == f"Image shape: {imageShape}; " \
                             f"# Metadata Keys: {keyCount}; " \
                             f"Version: {version}"

# Test that equality comparison is as expected
def testEquals(sampleNifti1, sample3DNifti, imageMetadataDict):
    # Test images with different headers
    assert BidsIncremental(sampleNifti1, imageMetadataDict) != \
           BidsIncremental(sample3DNifti, imageMetadataDict)

    # Test images with the same header, but different data
    newData = 1.1 * sampleNifti1.get_fdata()
    reversedNifti1 = nib.Nifti1Image(newData, sampleNifti1.affine,
                                     header=sampleNifti1.header)
    assert BidsIncremental(sampleNifti1, imageMetadataDict) != \
        BidsIncremental(reversedNifti1, imageMetadataDict)

    # Test different image metadata
    modifiedImageMetadata = deepcopy(imageMetadataDict)
    modifiedImageMetadata["subject"] = "newSubject"
    assert BidsIncremental(sampleNifti1, imageMetadataDict) != \
           BidsIncremental(sampleNifti1, modifiedImageMetadata)

    # Test different dataset metadata
    datasetMeta1 = {"Name": "Dataset_1", "BIDSVersion": "1.0"}
    datasetMeta2 = {"Name": "Dataset_2", "BIDSVersion": "2.0"}
    assert BidsIncremental(sampleNifti1, imageMetadataDict, datasetMeta1) != \
           BidsIncremental(sampleNifti1, imageMetadataDict, datasetMeta2)

# Test that image metadata dictionaries can be properly created by the class
def testImageMetadataDictCreation(imageMetadataDict):
    logger.debug("Subject: %s", imageMetadataDict["subject"])
    createdDict = BidsIncremental.createImageMetadataDict(
                    subject=imageMetadataDict["subject"],
                    task=imageMetadataDict["task"],
                    suffix=imageMetadataDict["suffix"],
                    repetitionTime=imageMetadataDict["RepetitionTime"],
                    echoTime=imageMetadataDict["EchoTime"])

    for key in createdDict.keys():
        assert createdDict.get(key) == imageMetadataDict.get(key)

# Test that invalid dataset.json fields are rejected and valid ones are accepted
def testDatasetMetadata(sampleNifti1, imageMetadataDict):
    # Test invalid dataset metadata
    with pytest.raises(ValidationError):
        BidsIncremental(image=sampleNifti1,
                        imageMetadata=imageMetadataDict,
                        datasetMetadata={"random_field": "doesnt work"})

    # Test valid dataset metadata
    dataset_name = "Test dataset"
    bidsInc = BidsIncremental(image=sampleNifti1,
                              imageMetadata=imageMetadataDict,
                              datasetMetadata={"Name": dataset_name,
                                               "BIDSVersion": "1.0"})
    assert bidsInc.datasetName() == dataset_name


# Test that extracting metadata from the BIDS-I using its provided API returns
# the correct values
def testMetadataOutput(validBidsI, imageMetadataDict):
    with pytest.raises(ValueError):
        validBidsI.getEntity("InvalidEntityName")

    # Data type - always 'func' currently
    assert validBidsI.dataType() == "func"
    # Entities
    assert validBidsI.getEntity('subject') == imageMetadataDict["subject"]
    assert validBidsI.getEntity('task') == imageMetadataDict["task"]
    # Suffix
    assert validBidsI.suffix() == imageMetadataDict["suffix"]


# Test that the BIDS-I properly parses BIDS fields present in a DICOM
# ProtocolName header field
def testParseProtocolName():
    # ensure nothing spurious is found in strings without BIDS fields
    assert BidsIncremental.metadataFromProtocolName("") == {}
    assert BidsIncremental.metadataFromProtocolName("this ain't bids") == {}
    assert BidsIncremental.metadataFromProtocolName("nor_is_this") == {}
    assert BidsIncremental.metadataFromProtocolName("still-aint_it") == {}

    protocolName = "func_ses-01_task-story_run-01"
    expectedValues = {'session': '01', 'task': 'story', 'run': '01'}

    parsedValues = BidsIncremental.metadataFromProtocolName(protocolName)

    for key, expectedValue in expectedValues.items():
        assert parsedValues[key] == expectedValue


# Test adding BIDS-I entities API works as expected
def testAddEntity(validBidsI):
    with pytest.raises(ValueError):
        validBidsI.addEntity("nonentity", "value")

    entityName = "acquisition"
    newValue = "newValue"
    originalValue = validBidsI.getEntity(entityName)

    validBidsI.addEntity(entityName, newValue)
    assert validBidsI.getEntity(entityName) == newValue

    validBidsI.addEntity(entityName, originalValue)
    assert validBidsI.getEntity(entityName) == originalValue


# Test removing BIDS-I entities API works as expected
def testRemoveEntity(validBidsI):
    with pytest.raises(ValueError):
        validBidsI.removeEntity("nonentity")

    entityName = "acquisition"
    originalValue = validBidsI.getEntity(entityName)

    validBidsI.removeEntity(entityName)
    assert validBidsI.getEntity(entityName) is None

    validBidsI.addEntity(entityName, originalValue)
    assert validBidsI.getEntity(entityName) == originalValue


# Test that the BIDS-I interface methods for extracting internal NIfTI data
# return the correct values
def testQueryNifti(validBidsI):
    # Image data
    assert np.array_equal(validBidsI.imageData(),
                          validBidsI.image.get_fdata())

    # Header Data
    queriedHeader = validBidsI.imageHeader()
    exactHeader = validBidsI.image.header

    # Compare full image header
    assert queriedHeader.keys() == exactHeader.keys()
    for (field, queryValue) in queriedHeader.items():
        exactValue = exactHeader.get(field)
        if queryValue.dtype.char == 'S':
            assert queryValue == exactValue
        else:
            assert np.allclose(queryValue, exactValue, atol=0.0, equal_nan=True)

    # Compare Header field: Dimensions
    FIELD = "dim"
    assert np.array_equal(queriedHeader.get(FIELD), exactHeader.get(FIELD))


# Test that constructing BIDS-compatible filenames from internal metadata
# returns the correct filenames
def testFilenameConstruction(validBidsI, imageMetadataDict):
    """
    General format:
    sub-<label>[_ses-<label>]_task-<label>[_acq-<label>] [_ce-<label>]
        [_dir-<label>][_rec-<label>][_run-<index>]
        [_echo-<index>]_<contrast_label >.ext
    """
    baseFilename = "sub-{sub}_ses-{ses}_task-{task}_run-{run}_{suf}".format(
        sub=imageMetadataDict["subject"],
        ses=imageMetadataDict["session"],
        task=imageMetadataDict["task"],
        run=imageMetadataDict["run"],
        suf=imageMetadataDict["suffix"])

    assert baseFilename + ".nii" == \
        validBidsI.makeBidsFileName(BidsFileExtension.IMAGE)
    assert baseFilename + ".json" == \
        validBidsI.makeBidsFileName(BidsFileExtension.METADATA)


# Test that the hypothetical path for the BIDS-I if it were in an archive is
# correct based on the metadata within it
def testArchivePathConstruction(validBidsI, imageMetadataDict):
    session = "01"
    validBidsI.addEntity("session", session)

    assert validBidsI.dataDirPath() == \
        "/sub-{}/ses-{}/func/".format(imageMetadataDict["subject"], session)

    validBidsI.removeEntity("session")


# Test that writing the BIDS-I to disk returns a properly formatted BIDS archive
# in the correct location with all the data in the BIDS-I
def testDiskOutput(validBidsI):
    # Write the archive
    directoryPath = tempfile.gettempdir()
    validBidsI.writeToArchive(directoryPath)

    # Validate the BIDS-compliance of each path (relative to dataset root) of
    # every file in the archive
    datasetPath = os.path.join(directoryPath, validBidsI.datasetName(), "")
    validator = BIDSValidator()
    for dirPath, _, filenames in os.walk(datasetPath):
        for f in filenames:
            fname = os.path.join(dirPath, f)
            fname = fname.replace(datasetPath, os.path.sep)
            assert validator.is_bids(fname)

    # Cleanup temp directory if test succeeded; leave for inspection otherwise
    shutil.rmtree(datasetPath)


# Test serialization results in equivalent BIDS-I object
def testSerialization(validBidsI):
    # Pickle the object
    pickledBuf = io.BytesIO()
    pickle.dump(validBidsI, pickledBuf)

    # Unpickle the object
    pickledBuf.seek(0)
    unpickled = pickle.load(pickledBuf)

    # Compare equality
    assert unpickled == validBidsI
