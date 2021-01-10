from copy import deepcopy
import io
import logging
import os
import pickle
import shutil
import tempfile

from bids_validator import BIDSValidator
from bids.layout.writing import build_path as bids_build_path
import pytest
import nibabel as nib
import numpy as np

from rtCommon.bidsIncremental import BidsIncremental
from rtCommon.bidsCommon import (
    getNiftiData,
    BidsFileExtension,
    BIDS_DIR_PATH_PATTERN,
    BIDS_FILE_PATTERN,
    metadataFromProtocolName,
)
from rtCommon.errors import ValidationError

logger = logging.getLogger(__name__)


# Test that construction fails for image metadata missing required fields
def testInvalidConstruction(sample2DNifti, sample4DNifti1, imageMetadata):
    # Test empty image
    with pytest.raises(ValidationError):
        BidsIncremental(image=None,
                        imageMetadata=imageMetadata)

    # Test 2-D image
    with pytest.raises(ValidationError):
        BidsIncremental(image=sample2DNifti,
                        imageMetadata=imageMetadata)

    # Test incomplete metadata
    protocolName = imageMetadata.pop("ProtocolName")
    for key in BidsIncremental.REQUIRED_IMAGE_METADATA:
        value = imageMetadata.pop(key)

        assert not BidsIncremental.isCompleteImageMetadata(imageMetadata)
        with pytest.raises(ValidationError):
            BidsIncremental(image=sample4DNifti1,
                            imageMetadata=imageMetadata)

        imageMetadata[key] = value
    imageMetadata["ProtocolName"] = protocolName

    # Test too-large repetition and echo times
    for key in ["RepetitionTime", "EchoTime"]:
        original = imageMetadata[key]
        imageMetadata[key] = 10**6

        with pytest.raises(ValidationError):
            BidsIncremental(image=sample4DNifti1,
                            imageMetadata=imageMetadata)

        imageMetadata[key] = original

    # Test non-image object
    with pytest.raises(ValidationError):
        BidsIncremental(image="definitely not an image",
                        imageMetadata=imageMetadata)


# Test that valid arguments produce a BIDS incremental
def testValidConstruction(sample3DNifti1, sample3DNifti2,
                          sample4DNifti1, sampleNifti2, imageMetadata):
    # 3-D should be promoted to 4-D
    assert BidsIncremental(sample3DNifti1, imageMetadata) is not None
    assert BidsIncremental(sample3DNifti2, imageMetadata) is not None

    # Both Nifti1 and Nifti2 images should work
    assert BidsIncremental(sample4DNifti1, imageMetadata) is not None
    assert BidsIncremental(sampleNifti2, imageMetadata) is not None

    # If the metadata provides a RepetitionTime or EchoTime that works without
    # adjustment, the construction should still work
    repetitionTimeKey = "RepetitionTime"
    original = imageMetadata[repetitionTimeKey]
    imageMetadata[repetitionTimeKey] = 1.5
    assert BidsIncremental(sample4DNifti1, imageMetadata) is not None
    imageMetadata[repetitionTimeKey] = original


# Test that the provided image metadata dictionary takes precedence over the
# metadata parsed from the protocol name, if any
def testConstructionMetadataPrecedence(sample4DNifti1, imageMetadata):
    assert imageMetadata.get('ProtocolName', None) is not None
    metadata = metadataFromProtocolName(imageMetadata['ProtocolName'])
    assert len(metadata) > 0

    assert metadata.get('run', None) is not None
    newRunNumber = int(metadata['run']) + 1
    imageMetadata['run'] = newRunNumber
    assert metadata['run'] != imageMetadata['run']

    incremental = BidsIncremental(sample4DNifti1, imageMetadata)
    assert incremental.getMetadataField('run') == newRunNumber


# Test that the string output of the BIDS-I is as expected
def testStringOutput(validBidsI):
    imageShape = str(validBidsI.imageDimensions)
    keyCount = len(validBidsI._imgMetadata.keys())
    version = validBidsI.version
    assert str(validBidsI) == f"Image shape: {imageShape}; " \
                              f"# Metadata Keys: {keyCount}; " \
                              f"Version: {version}"


# Test that equality comparison is as expected
def testEquals(sample4DNifti1, sample3DNifti1, imageMetadata):
    # Test images with different headers
    assert BidsIncremental(sample4DNifti1, imageMetadata) != \
           BidsIncremental(sample3DNifti1, imageMetadata)

    # Test images with the same header, but different data
    newData = 2 * getNiftiData(sample4DNifti1)
    reversedNifti1 = nib.Nifti1Image(newData, sample4DNifti1.affine,
                                     header=sample4DNifti1.header)
    assert BidsIncremental(sample4DNifti1, imageMetadata) != \
        BidsIncremental(reversedNifti1, imageMetadata)

    # Test different image metadata
    modifiedImageMetadata = deepcopy(imageMetadata)
    modifiedImageMetadata["subject"] = "newSubject"
    assert BidsIncremental(sample4DNifti1, imageMetadata) != \
           BidsIncremental(sample4DNifti1, modifiedImageMetadata)

    # Test different dataset metadata
    datasetMeta1 = {"Name": "Dataset_1", "BIDSVersion": "1.0"}
    datasetMeta2 = {"Name": "Dataset_2", "BIDSVersion": "2.0"}
    assert BidsIncremental(sample4DNifti1, imageMetadata, datasetMeta1) != \
           BidsIncremental(sample4DNifti1, imageMetadata, datasetMeta2)


# Test that image metadata dictionaries can be properly created by the class
def testImageMetadataDictCreation(imageMetadata):
    createdDict = BidsIncremental.createImageMetadataDict(
                    subject=imageMetadata["subject"],
                    task=imageMetadata["task"],
                    suffix=imageMetadata["suffix"],
                    repetitionTime=imageMetadata["RepetitionTime"],
                    echoTime=imageMetadata["EchoTime"])

    for key in createdDict.keys():
        assert createdDict.get(key) == imageMetadata.get(key)


# Test that invalid dataset.json fields are rejected and valid ones are accepted
def testDatasetMetadata(sample4DNifti1, imageMetadata):
    # Test invalid dataset metadata
    with pytest.raises(ValidationError):
        BidsIncremental(image=sample4DNifti1,
                        imageMetadata=imageMetadata,
                        datasetMetadata={"random_field": "doesnt work"})

    # Test valid dataset metadata
    dataset_name = "Test dataset"
    bidsInc = BidsIncremental(image=sample4DNifti1,
                              imageMetadata=imageMetadata,
                              datasetMetadata={"Name": dataset_name,
                                               "BIDSVersion": "1.0"})
    assert bidsInc.datasetName() == dataset_name


# Test that extracting metadata from the BIDS-I using its provided API returns
# the correct values
def testMetadataOutput(validBidsI, imageMetadata):
    with pytest.raises(ValueError):
        validBidsI.getMetadataField("InvalidEntityName", strict=True)
    assert validBidsI.getMetadataField("InvalidEntityName") is None

    # Data type - always 'func' currently
    assert validBidsI.dataType() == "func"
    # Entities
    for entity in ['subject', 'task']:
        assert validBidsI.getMetadataField(entity) == imageMetadata[entity]
    # Suffix
    assert validBidsI.suffix() == imageMetadata["suffix"]


# Test setting BIDS-I metadata API works as expected
def testSetMetadata(validBidsI):
    # Test non-official BIDS entity fails with strict
    with pytest.raises(ValueError):
        validBidsI.setMetadataField("nonentity", "value", strict=True)

    # Non-official BIDS entity succeeds without strict
    validBidsI.setMetadataField("nonentity", "value", strict=False)
    assert validBidsI.getMetadataField("nonentity", strict=False) == "value"
    validBidsI.removeMetadataField("nonentity", strict=False)

    # None field is invalid
    with pytest.raises(ValueError):
        validBidsI.setMetadataField(None, None)

    entityName = "acquisition"
    newValue = "newValue"
    originalValue = validBidsI.getMetadataField(entityName)

    validBidsI.setMetadataField(entityName, newValue)
    assert validBidsI.getMetadataField(entityName) == newValue

    validBidsI.setMetadataField(entityName, originalValue)
    assert validBidsI.getMetadataField(entityName) == originalValue


# Test removing BIDS-I metadata API works as expected
def testRemoveMetadata(validBidsI):
    # Fail for entities that don't exist
    with pytest.raises(ValueError):
        validBidsI.removeMetadataField("nonentity", strict=True)

    # Fail for entities that are required to be in the dictionary
    with pytest.raises(ValueError):
        validBidsI.removeMetadataField("subject")

    entityName = "acquisition"
    originalValue = validBidsI.getMetadataField(entityName)

    validBidsI.removeMetadataField(entityName)
    assert validBidsI.getMetadataField(entityName) is None

    validBidsI.setMetadataField(entityName, originalValue)
    assert validBidsI.getMetadataField(entityName) == originalValue


# Test that the BIDS-I interface methods for extracting internal NIfTI data
# return the correct values
def testQueryNifti(validBidsI):
    # Image data
    queriedData = validBidsI.imageData()
    exactData = getNiftiData(validBidsI.image)
    assert np.array_equal(queriedData, exactData), "{} elements not equal" \
        .format(np.sum(np.where(queriedData != exactData)))

    # Header Data
    queriedHeader = validBidsI.imageHeader
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
def testFilenameConstruction(validBidsI, imageMetadata):
    """
    General format:
    sub-<label>[_ses-<label>]_task-<label>[_acq-<label>] [_ce-<label>]
        [_dir-<label>][_rec-<label>][_run-<index>]
        [_echo-<index>]_<contrast_label >.ext
    """
    baseFilename = bids_build_path(imageMetadata, BIDS_FILE_PATTERN)

    assert baseFilename + ".nii" == \
        validBidsI.makeBidsFileName(BidsFileExtension.IMAGE)
    assert baseFilename + ".json" == \
        validBidsI.makeBidsFileName(BidsFileExtension.METADATA)


# Test that the hypothetical path for the BIDS-I if it were in an archive is
# correct based on the metadata within it
def testArchivePathConstruction(validBidsI, imageMetadata):
    assert validBidsI.dataDirPath() == \
        '/' + bids_build_path(imageMetadata, BIDS_DIR_PATH_PATTERN) + '/'


# Test that writing the BIDS-I to disk returns a properly formatted BIDS archive
# in the correct location with all the data in the BIDS-I
def testDiskOutput(validBidsI):
    # Write the archive
    datasetRoot = os.path.join(tempfile.gettempdir(), "bids-pytest-dataset")
    validBidsI.writeToArchive(datasetRoot)

    # Validate the BIDS-compliance of each path (relative to dataset root) of
    # every file in the archive
    validator = BIDSValidator()
    for dirPath, _, filenames in os.walk(datasetRoot):
        for f in filenames:
            fname = os.path.join(dirPath, f)
            fname = fname.replace(datasetRoot, "")
            assert validator.is_bids(fname)

    # Cleanup temp directory if test succeeded; leave for inspection otherwise
    shutil.rmtree(datasetRoot)


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
