import io
import logging
import os
import pickle
import re
import shutil
import tempfile

from bids_validator import BIDSValidator
import pytest
import numpy as np

from rtCommon.bidsIncremental import BidsIncremental
from rtCommon.bidsCommon import BidsFileExtension as BidsFileExtension
from rtCommon.errors import ValidationError

logger = logging.getLogger(__name__)

# Permutations that the constructor must validate
nonePermutations = [{"subject": None, "task": None, "suffix": None},
                    {"subject": "test", "task": None, "suffix": None},
                    {"subject": "test", "task": "test", "suffix": None}]


@pytest.mark.parametrize("argDict", nonePermutations)
def testNullConstruction(sampleNiftiImage, imageMetadataDict, argDict):
    # Test empty image
    with pytest.raises(ValidationError):
        BidsIncremental(image=None,
                        subject=imageMetadataDict["subject"],
                        task=imageMetadataDict["task"],
                        suffix=imageMetadataDict["suffix"])

        # Test empty required fields
        for p in nonePermutations:
            with pytest.raises(ValidationError) as excinfo:
                BidsIncremental(image=sampleNiftiImage,
                                subject=p["subject"],
                                task=p["task"],
                                suffix=p["suffix"],
                                imgMetadata=imageMetadataDict)
            assert "Image" not in str(excinfo.value)


def testValidConstruction(sampleNiftiImage, imageMetadataDict):
    bidsInc = BidsIncremental(image=sampleNiftiImage,
                              subject=imageMetadataDict["subject"],
                              task=imageMetadataDict["task"],
                              suffix=imageMetadataDict["suffix"],
                              imgMetadata=imageMetadataDict)
    assert bidsInc is not None


# Test that invalid dataset.json fields are rejected and valid ones are accepted
def testDatasetMetadata(sampleNiftiImage, imageMetadataDict):
    # Test invalid dataset metadata
    with pytest.raises(ValidationError):
        BidsIncremental(image=sampleNiftiImage,
                        subject=imageMetadataDict["subject"],
                        task=imageMetadataDict["task"],
                        suffix=imageMetadataDict["suffix"],
                        datasetMetadata={"random_field": "doesnt work"},
                        imgMetadata=imageMetadataDict)

    # Test valid dataset metadata
    dataset_name = "Test dataset"
    bidsInc = BidsIncremental(image=sampleNiftiImage,
                              subject=imageMetadataDict["subject"],
                              task=imageMetadataDict["task"],
                              suffix=imageMetadataDict["suffix"],
                              datasetMetadata={"Name": dataset_name,
                                               "BIDSVersion": "1.0"},
                              imgMetadata=imageMetadataDict)
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
    protocolName = "func_ses-01_task-story_run-01"
    expectedValues = {'session': '01', 'task': 'story', 'run': '01'}

    parsedValues = BidsIncremental.parseBidsFieldsFromProtocolName(protocolName)

    for key, expectedValue in expectedValues.items():
        assert parsedValues[key] == expectedValue


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
