import logging
import os
import shutil
import tempfile

from rtCommon import bidsLibrary as bl
from rtCommon.bidsArchive import BidsArchive
from rtCommon import bidsCommon

logger = logging.getLogger(__name__)


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


# Test images are correctly appended to an empty archive
def testEmptyArchiveAppend(validBidsI, imageMetadataDict, tmpdir):
    # Create in root with no BIDS-I, then append to make non-empty archive
    datasetRoot = os.path.join(tmpdir, testEmptyArchiveAppend.__name__)
    archive = BidsArchive(datasetRoot)
    archive.appendIncremental(validBidsI)
    assert not archive.isEmpty()

    # Compare metadata reported by PyBids to metadata we expect has been written
    bidsLayout = archive.dataset.data
    metadata = {}
    for f in bidsLayout.get(return_type='filename'):
        if not bidsCommon.isNiftiPath(f):
            continue
        metadata.update(bidsLayout.get_metadata(f, include_entities=True))
    for key, value in metadata.items():
        niftiValue = imageMetadataDict.get(key, None)
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
            assert False, f"{niftiValue}, type {type(niftiValue)} != {value}, "\
                          f"type {type(value)}"


# Test images are correctly appended to an archive with just a 3-D image in it
def test3DAppend(bidsArchive3D):
    pass


# Test images are correctly appended to an archive with a 4-D sequence in it
def test4DAppend():
    pass


# Test stripping an image off from a BIDS archive works as expected
def testStripImage():
    pass
