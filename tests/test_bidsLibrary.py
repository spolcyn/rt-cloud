import os
import shutil
import tempfile

from rtCommon import bidsLibrary as bl
from rtCommon.bidsArchive import BidsArchive


# Test metadata is correctly extracted from a DICOM to public and private
# dictionaries by ensuring a sample of public keys have the right value
def testMetadataExtraction(dicomImage, dicomMetadataSample):
    public, private = bl.getMetadata(dicomImage)
    for field, value in dicomMetadataSample.items():
        assert public.get(field) == str(value)

    # TODO: Also check private keys
    pass


# Test creating archive without a path
def testEmptyArchiveCreation():
    datasetRoot = os.path.join(tempfile.gettempdir(), "bids-archive")
    assert BidsArchive(datasetRoot) is not None

    # Clean up
    shutil.rmtree(datasetRoot)


# Test images are correctly appended to an empty archive
def testEmptyArchiveAppend(validBidsI, imageMetadataDict):
    # Test creation in root with no BIDS-I, then append to get a non-empty
    archive = BidsArchive(os.path.join(tempfile.gettempdir(), "bids-archive"))
    archive.appendIncremental(validBidsI)
    assert not archive.isEmpty()

    # Verify all BIDS-I metadata is discovered by the BIDS archive dataset

    # Clean up
    shutil.rmtree(datasetRoot)


# Test images are correctly appended to an archive with just a 3-D image in it
def test3DAppend():
    pass


# Test images are correctly appended to an archive with a 4-D sequence in it
def test4DAppend():
    pass


# Test stripping an image off from a BIDS archive works as expected
def testStripImage():
    pass
