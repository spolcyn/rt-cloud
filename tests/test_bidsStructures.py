import os
import pytest
import tempfile
import time

import rtCommon.bidsStructures as bs
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
def validMetadataDict():
    return {'sub': '01', 'task': 'aTask', 'contrast_label': 'bold'}

def test_createBidsIncremental(sampleNiftiImage, validMetadataDict):
    # BIDS-I constructor must have non-None arguments
    with pytest.raises(ValidationError):
        bs.BidsIncremental(None, None)

    # BIDS-I constructor must have metadata provided
    with pytest.raises(ValidationError):
        bs.BidsIncremental(sampleNiftiImage, None)

    # BIDS-I constructor must have all required metadata provided
    with pytest.raises(ValidationError):
        requiredFields = dict.fromkeys(['sub', 'task', 'contrast_label'], None)
        bs.BidsIncremental(sampleNiftiImage, None)

    # Build valid BIDS-I
    assert bs.BidsIncremental(sampleNiftiImage, validMetadataDict) is not None
