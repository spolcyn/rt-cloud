import logging

from rtCommon import bidsLibrary as bl

logger = logging.getLogger(__name__)


# Test metadata is correctly extracted from a DICOM to public and private
# dictionaries by ensuring a sample of public keys have the right value
def testMetadataExtraction(dicomImage, dicomMetadataSample):
    public, private = bl.getMetadata(dicomImage)
    for field, value in dicomMetadataSample.items():
        assert public.get(field) == str(value)

    # TODO(spolcyn): Also check private keys
    pass
