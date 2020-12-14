from rtCommon import bidsLibrary as bl


# Test metadata is correctly extracted from a DICOM to public and private
# dictionaries by ensuring a sample of public keys have the right value
def testMetadataExtraction(dicomImage, dicomMetadataSample):
    public, private = bl.getMetadata(dicomImage)
    for field, value in dicomMetadataSample.items():
        assert public.get(field) == str(value)

    # TODO: Also check private keys
    pass


# Test images are correctly appended to an empty archive
def testEmptyArchiveAppend():
    pass


# Test images are correctly appended to an archive with just a 3-D image in it
def test3DAppend():
    pass


# Test images are correctly appended to an archive with a 4-D sequence in it
def test4DAppend():
    pass


# Test stripping an image off from a BIDS archive works as expected
def testStripImage():
    pass
