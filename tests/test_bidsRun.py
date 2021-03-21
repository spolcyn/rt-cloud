import logging

import nibabel as nib
import pytest

from rtCommon.bidsIncremental import BidsIncremental
from rtCommon.bidsRun import BidsRun
from rtCommon.errors import MetadataMismatchError

logger = logging.getLogger(__name__)

# Test equality check is correct
def testEq(validBidsI):
    run1 = BidsRun()
    run2 = BidsRun()

    run1.appendIncremental(validBidsI)
    assert run1 != run2

    run2.appendIncremental(validBidsI)
    assert run1 == run2

    run2._entities['subject'] = "new_subect"
    assert run1 != run2

# Test numIncrementals output is correct
def testNumVols(validBidsI):

    run = BidsRun()
    assert run.numIncrementals() == 0

    NUM_APPENDS = 20
    for i in range(NUM_APPENDS):
        run.appendIncremental(validBidsI)
        assert run.numIncrementals() == i + 1

# Test out of bounds values for get incremental
def testGetOutOfBounds(validBidsI):
    run = BidsRun()

    NUM_APPENDS = 10
    for i in range(NUM_APPENDS):
        run.appendIncremental(validBidsI)

    with pytest.raises(IndexError):
        run.getIncremental(NUM_APPENDS)
        run.getIncremental(NUM_APPENDS + 1)
        run.getIncremental(-1 * NUM_APPENDS)
        run.getIncremental(-1 * NUM_APPENDS - 1)


# Test get and append
def testGetAppendIncremental(validBidsI):
    run = BidsRun()

    run.appendIncremental(validBidsI)
    assert run.getIncremental(0) == validBidsI

    NUM_APPENDS = 20
    for i in range(1, NUM_APPENDS):
        validBidsI.setMetadataField('append_num', i)
        run.appendIncremental(validBidsI)
        assert run.getIncremental(i).getMetadataField('append_num') == i
        assert run.getIncremental(-1).getMetadataField('append_num') == i
        assert run.numIncrementals() == i + 1

# Test construction
def testConstruction(validBidsI, sampleBidsEntities):
    runWithoutEntities = BidsRun()
    assert runWithoutEntities is not None
    assert len(runWithoutEntities.getRunEntities()) == 0

    runWithEntities = BidsRun(**sampleBidsEntities)
    assert runWithEntities is not None
    assert runWithEntities.getRunEntities() == sampleBidsEntities

# Test append correctly sets entities
def testAppendSetEntities(validBidsI, sampleBidsEntities):
    run = BidsRun()
    run.appendIncremental(validBidsI)
    assert run.getRunEntities() == sampleBidsEntities

# Test append works correctly if entities are set but incremental list is empty
def testAppendEmptyIncrementals(validBidsI, sampleBidsEntities):
    run = BidsRun(**sampleBidsEntities)
    run.appendIncremental(validBidsI)
    assert run.numIncrementals() == 1

# Test append doesn't work with mismatched entities
def testAppendConflictingEntities(validBidsI):
    differentBidsInc = BidsIncremental(validBidsI.image,
                                       validBidsI.imageMetadata)
    differentBidsInc.setMetadataField("subject", "new-subject")

    run = BidsRun()
    run.appendIncremental(validBidsI)
    with pytest.raises(MetadataMismatchError):
        run.appendIncremental(differentBidsInc)

# Test append doesn't work if NIfTI headers don't match
def testAppendConflictingNiftiHeaders(sample4DNifti1, imageMetadata):
    bidsInc1 = BidsIncremental(sample4DNifti1, imageMetadata)

    # Change the pixel dimensions (zooms) to make the image append-incompatible
    image2 = nib.Nifti1Image(sample4DNifti1.dataobj,
                             sample4DNifti1.affine,
                             sample4DNifti1.header)
    new_data_shape = tuple(i * 2 for i in image2.header.get_zooms())
    image2.header.set_zooms(new_data_shape)
    bidsInc2 = BidsIncremental(image2, imageMetadata)

    run = BidsRun()
    run.appendIncremental(bidsInc1)
    with pytest.raises(MetadataMismatchError):
        run.appendIncremental(bidsInc2)

    # Append should work with unsafeAppend turned on
    numIncrementalsBefore = run.numIncrementals()
    run.appendIncremental(bidsInc2, unsafeAppend=True)
    assert run.numIncrementals() == (numIncrementalsBefore + 1)
