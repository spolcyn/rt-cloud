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


def archiveHasMetadata(archive: BidsArchive, metadata: dict) -> bool:
    """
    Test if archive's metadata matches provided metadata dict
    """

    # Compare metadata reported by PyBids to metadata we expect has been written
    bidsLayout = archive.dataset.data
    archiveMetadata = {}
    for f in bidsLayout.get(return_type='filename'):
        if not bidsCommon.isNiftiPath(f):
            continue
        archiveMetadata.update(bidsLayout.get_metadata(f, include_entities=True))
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
            logger.debug(f"{niftiValue}, type {type(niftiValue)} != {value}, " \
                         f"type {type(value)}")
            return False

    return True


# Test images are correctly appended to an empty archive
def testEmptyArchiveAppend(validBidsI, imageMetadataDict, tmpdir):
    # Create in root with no BIDS-I, then append to make non-empty archive
    datasetRoot = os.path.join(tmpdir, testEmptyArchiveAppend.__name__)
    archive = BidsArchive(datasetRoot)
    archive.appendIncremental(validBidsI)
    assert not archive.isEmpty()

    assert archiveHasMetadata(archive, imageMetadataDict)


# Test images are correctly appended to an archive with just a 3-D image in it
def test3DAppend(bidsArchive3D, validBidsI, imageMetadataDict):
    bidsArchive3D.appendIncremental(validBidsI)
    assert archiveHasMetadata(bidsArchive3D, imageMetadataDict)


# Test images are correctly appended to an archive with a single 4-D image in it
def test4DAppend(bidsArchive4D, validBidsI, imageMetadataDict):
    bidsArchive4D.appendIncremental(validBidsI)
    assert archiveHasMetadata(bidsArchive4D, imageMetadataDict)

# Test images are correctly appended to an archive with a 4-D sequence in it
def testSequenceAppend(bidsArchive4D, validBidsI, imageMetadataDict):
    NUM_APPENDS = 2
    BIDSI_LENGTH = 2

    for i in range(NUM_APPENDS):
        bidsArchive4D.appendIncremental(validBidsI)

    # TODO(spolcyn): Make the image path not hardcoded
    image = bidsArchive4D.getImage("/sub-01/ses-01/func/sub-01_ses-01_task-story_run-1_bold.nii")
    assert len(image.get_fdata().shape) == 4
    assert image.get_fdata().shape[3] == (BIDSI_LENGTH * (1 + NUM_APPENDS))

    assert archiveHasMetadata(bidsArchive4D, imageMetadataDict)

# Test appending a new subject (and thus creating a new directory) to a
# non-empty BIDS Archive
def testAppendNewSubject(bidsArchive4D):


# Test stripping an image off from a BIDS archive works as expected
def testStripImage():
    pass
