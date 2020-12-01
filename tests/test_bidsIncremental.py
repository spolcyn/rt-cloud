import io
import os
import pickle
import tempfile
import time

import pytest

from rtCommon.bidsIncremental import BidsIncremental
from rtCommon.errors import ValidationError
from rtCommon.imageHandling import convertDicomFileToNifti, readNifti
from tests.common import test_inputDir, test_dicomFile


@pytest.fixture
def sampleNiftiImage():
    dicomPath = os.path.join(os.path.dirname(__file__),
                             test_inputDir,
                             test_dicomFile)
    niftiPath = os.path.join(tempfile.gettempdir(), "test.nii")
    convertDicomFileToNifti(dicomPath, niftiPath)

    # 'Yield' will return the Nifti image now, and delay the running of
    # subsequent code (i.e., the removal of the Nifti image on disk) until the
    # end-of-test tear-down is being done
    # This ensures the test doesn't create clutter, but is able to access the
    # necessary data during the test
    yield readNifti(niftiPath)

    os.remove(niftiPath)

@pytest.fixture
def requiredMetadataDict():
    return {'subject': '01', 'task': 'aTask', 'suffix': 'bold'}

@pytest.fixture
def validBidsIncremental(sampleNiftiImage, requiredMetadataDict):
    return BidsIncremental(image=sampleNiftiImage,
                           subject=requiredMetadataDict["subject"],
                           task=requiredMetadataDict["task"],
                           suffix=requiredMetadataDict["suffix"])

nonePermutations = [{"subject": None, "task": None, "suffix": None},
                    {"subject": "test", "task": None, "suffix": None},
                    {"subject": "test", "task": "test", "suffix": None}]

@pytest.mark.parametrize("argDict", nonePermutations)
def testNullConstruction(sampleNiftiImage, requiredMetadataDict, argDict):
    # Test empty image
    with pytest.raises(ValidationError):
        BidsIncremental(image=None,
                        subject=requiredMetadataDict["subject"],
                        task=requiredMetadataDict["task"],
                        suffix=requiredMetadataDict["suffix"])

        # Test empty required fields
        for p in nonePermutations:
            with pytest.raises(ValidationError) as excinfo:
                BidsIncremental(image=sampleNiftiImage,
                                subject=p["subject"],
                                task=p["task"],
                                suffix=p["suffix"])
            assert "Image" not in str(excinfo.value)

def testValidConstruction(sampleNiftiImage, requiredMetadataDict):
    bidsInc = BidsIncremental(image=sampleNiftiImage,
                              subject=requiredMetadataDict["subject"],
                              task=requiredMetadataDict["task"],
                              suffix=requiredMetadataDict["suffix"])
    assert bidsInc is not None

def testDatasetMetadata(sampleNiftiImage, requiredMetadataDict):
    # Test invalid dataset metadata
    with pytest.raises(ValidationError):
        BidsIncremental(image=sampleNiftiImage,
                        subject=requiredMetadataDict["subject"],
                        task=requiredMetadataDict["task"],
                        suffix=requiredMetadataDict["suffix"],
                        datasetMetadata={"random_field": "doesnt work"})

    # Test valid dataset metadata
    bidsInc = BidsIncremental(image=sampleNiftiImage,
                              subject=requiredMetadataDict["subject"],
                              task=requiredMetadataDict["task"],
                              suffix=requiredMetadataDict["suffix"],
                              datasetMetadata={"Name": "Test dataset",
                                               "BIDSVersion": "1.0"})
    assert bidsInc is not None



# Test that extracting metadata from the BIDS-I using its provided API returns
# the correct values
def testMetadataOutput(validBidsIncremental):
    with pytest.raises(ValueError):
        validBidsIncremental.getEntity("InvalidEntityName")

    # TODO

    pass

# Test that the BIDS-I properly parses BIDS fields present in a DICOM
# ProtocolName header field
def testParseProtocolName():
    protocolName = "func_ses-01_task-story_run-01"
    expectedValues = {'ses': '01', 'task': 'story', 'run': '01'}

    parsedValues = BidsIncremental.parseBidsFieldsFromProtocolName(protocolName)

    for key, expectedValue in expectedValues.items():
        assert parsedValues[key] == expectedValue


# Test that constructing BIDS-compatible filenames from internal metadata
# returns the correct filenames
def testFilenameConstruction():
    pass

# Test that the hypothetical path for the BIDS-I if it were in an archive is
# correct based on the metadata within it
def testArchivePathConstruction():
    pass

# Test that writing the BIDS-I to disk returns a properly formatted BIDS archive
# in the correct location with all the data in the BIDS-I
def testDiskOutput():
    pass

# Test serialization results in equivalent BIDS-I object
def testSerialization(validBidsIncremental):
    # Pickle the object
    pickledBuf = io.BytesIO()
    pickle.dump(validBidsIncremental, pickledBuf)

    # Unpickle the object
    pickledBuf.seek(0)
    unpickled = pickle.load(pickledBuf)

    # Compare equality
    assert unpickled == validBidsIncremental
