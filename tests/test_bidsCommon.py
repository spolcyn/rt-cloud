import logging

from rtCommon.bidsCommon import (
    getMetadata,
    loadBidsEntities,
    metadataFromProtocolName,
)

logger = logging.getLogger(__name__)


def testEntitiesDictGeneration():
    """
    Ensure entitity dictionary is loaded and parsed properly
    Expected dictionary format:
      key: Full entity name, all lowercase
      value: Dictionary with keys "entitity, format, description"
    """
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


# Test metadata is correctly extracted from a DICOM to public and private
# dictionaries by ensuring a sample of public keys have the right value
def testMetadataExtraction(dicomImage, dicomMetadataSample):
    public, private = getMetadata(dicomImage)
    for field, value in dicomMetadataSample.items():
        assert public.get(field) == str(value)

    # TODO(spolcyn): Also check private keys
    pass


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
