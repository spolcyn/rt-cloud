import logging

import numpy as np
import pytest

from rtCommon.bidsCommon import (
    adjustTimeUnits,
    getDicomMetadata,
    getNiftiData,
    isNiftiPath,
    isJsonPath,
    loadBidsEntities,
    metadataFromProtocolName,
)

logger = logging.getLogger(__name__)


# Test time units are adjusted correctly from DICOM (msec) to BIDS (sec)
def testTimeUnitAdjustment():
    rtKey = 'RepetitionTime'
    etKey = 'EchoTime'
    metadata = {}

    # Test values above max, but convertible
    rtConvertible = 1000
    etConvertible = 10
    metadata[rtKey] = rtConvertible
    metadata[etKey] = etConvertible

    adjustTimeUnits(metadata)
    assert metadata[rtKey] == rtConvertible / 1000.0
    assert metadata[etKey] == etConvertible / 1000.0

    # Test values above max, but convertible
    rtAboveMax = 1000 * 100 + 1
    etAboveMax = 1000 * 1 + 1
    metadata[rtKey] = rtAboveMax
    metadata[etKey] = etAboveMax

    with pytest.raises(ValueError):
        adjustTimeUnits(metadata)

    # Test values within max
    rtWithinMax = 50
    etWithinMax = .5
    metadata[rtKey] = rtWithinMax
    metadata[etKey] = etWithinMax

    adjustTimeUnits(metadata)
    assert metadata[rtKey] == rtWithinMax
    assert metadata[etKey] == etWithinMax

    # Test missing values
    metadata[rtKey] = None
    metadata[etKey] = None

    adjustTimeUnits(metadata)
    assert metadata[rtKey] is None
    assert metadata[etKey] is None


# Test metadata is correctly extracted from a DICOM to public and private
# dictionaries by ensuring a sample of public keys have the right value
def testMetadataExtraction(dicomImage, dicomMetadataSample):
    with pytest.raises(TypeError):
        getDicomMetadata("this isn't a pydicom dataset")

    # Test a sampling of field names and values extracted by hand
    public, private = getDicomMetadata(dicomImage)
    for field, value in dicomMetadataSample.items():
        assert public.get(field) == str(value)

    # TODO(spolcyn): Also check private keys


# Ensure entitity dictionary is loaded and parsed properly
# Expected dictionary format:
#   key: Full entity name, all lowercase
def testEntitiesDictGeneration():
    entities = loadBidsEntities()

    # Ensure entity count correct
    NUM_ENTITIES = 20  # Manually counted from the file entities.yaml
    assert len(entities) == NUM_ENTITIES

    # Ensure case correct
    for key in entities.keys():
        assert key.islower()

    # Ensure expected values are present for each entity
    expectedValueKeys = ["entity", "format", "description"]
    for valueDict in entities.values():
        for key in expectedValueKeys:
            assert key in valueDict.keys()

    # Check a sample of important keys are present
    importantKeySample = ["subject", "task", "session"]
    for key in importantKeySample:
        assert key in entities.keys()


# Test BIDS fields in a DICOM ProtocolName header field are properly parsed
def testParseProtocolName():
    # ensure nothing spurious is found in strings without BIDS fields
    assert metadataFromProtocolName("") == {}
    assert metadataFromProtocolName("this ain't bids") == {}
    assert metadataFromProtocolName("nor_is_this") == {}
    assert metadataFromProtocolName("still-aint_it") == {}

    protocolName = "func_ses-01_task-story_run-01"
    expectedValues = {'session': '01', 'task': 'story', 'run': '01'}

    parsedValues = metadataFromProtocolName(protocolName)

    for key, expectedValue in expectedValues.items():
        assert parsedValues[key] == expectedValue


# Test correct Nifti data is extracted
def testGetNiftiData(sample4DNifti1):
    extracted = getNiftiData(sample4DNifti1)
    fromRawDataobj = np.asanyarray(sample4DNifti1.dataobj,
                                   dtype=sample4DNifti1.dataobj.dtype)

    assert np.array_equal(extracted, fromRawDataobj)


# Test NIfTI paths are correctly identified
def testCheckNiftiPath():
    assert isNiftiPath("test.nii")
    assert isNiftiPath("test_test_test_test.run2.old.exp16.nii")
    assert not isNiftiPath("test.jpeg")
    assert not isNiftiPath("test.nii.jpeg")
    assert not isNiftiPath("test.nii.gz")


# Test JSON (aka BIDS sidecar metdata) paths are correctly identified
def testCheckJsonPath():
    assert isJsonPath("test.json")
    assert isJsonPath("test_test_test_test.run2.old.exp16.json")
    assert not isJsonPath("test.jpeg")
    assert not isJsonPath("test.json.jpeg")
    assert not isJsonPath("test.json.gz")
