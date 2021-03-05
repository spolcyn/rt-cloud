"""-----------------------------------------------------------------------------

dicomToBidsService.py

Eventually, this will implement conversion between DICOM and BIDS.

-----------------------------------------------------------------------------"""

def dicomToBidsinc(dicomImg: pydicom.dataset.Dataset) -> BidsIncremental:
    # TODO(spolcyn): Do this all in memory -- dicom2nifti is promising
    # Put extra metadata in sidecar JSON file
    # Currently, there are 4 disk operations:
    # 1) Read DICOM (by dcm2niix)
    # 2) Write NIfTI
    # 3) Read NIfTI
    # 4) Read DICOM (for metadata)

    # NOTE: This is not the final version of this method.
    # The conversion from DICOM to BIDS-I and gathering all required metadata
    # can be complex, as DICOM doesn't necessarily have the metadata required
    # for BIDS in it by default. Thus, another component will handle the logic
    # and error handling surrounding this.
    niftiImage = convertDicomImgToNifti(dicomImg)
    publicMeta, privateMeta = getMetadata(dicomImg)

    publicMeta.update(privateMeta)  # combine metadata dictionaries
    return BidsIncremental(image=niftiImage, imageMetadata=publicMeta)
