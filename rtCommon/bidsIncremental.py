"""-----------------------------------------------------------------------------

bidsIncremental.py

This script includes all of the functions that are needed to convert between
DICOM, BIDS-Incremental (BIDS-I), and BIDS formats.

-----------------------------------------------------------------------------"""

import logging
import nibabel as nib
import numpy as np
import os
import pydicom
import random
import re

from rtCommon.bidsStructures import BidsArchive, BidsIncremental
import rtCommon.bidsUtils as bidsUtils
from rtCommon.errors import ValidationError
from rtCommon.imageHandling import convertDicomImgToNifti

logger = logging.getLogger(__name__)

# Set to True to skip validating Nifti header when appending
DISABLE_NIFTI_HEADER_CHECK = False

# Set to True to skip validating metadata when appending
DISABLE_METADATA_CHECK = True


def getMetadata(dicomImg: pydicom.dataset.Dataset) -> (dict, dict):
    """
    Returns the public and private metadata from the provided DICOM image.

    Args:
        dicomImg: A pydicom object to read metadata from.
    Returns:
        Tuple of 2 dictionaries, the first containing the public metadata from
        the image and the second containing the private metadata.
    """
    if not isinstance(dicomImg, pydicom.dataset.Dataset):
        raise ValidationError("Expected pydicom.dataset.Dataset as argument")

    publicMeta = {}
    privateMeta = {}

    ignoredTags = ['Pixel Data']

    for elem in dicomImg:
        if elem.name in ignoredTags:
            continue

        cleanedKey = bidsUtils.makeDicomFieldBidsCompatible(elem.name)
        # in DICOM, public tags have even group numbers and private tags are odd
        # http://dicom.nema.org/dicom/2013/output/chtml/part05/chapter_7.html
        if elem.tag.is_private:
            privateMeta[cleanedKey] = str(elem.value)
        else:
            publicMeta[cleanedKey] = str(elem.value)

    return (publicMeta, privateMeta)


def dicomToBidsinc(dicomImg: pydicom.dataset.Dataset) -> BidsIncremental:
    # TODO(spolcyn): Do this all in memory -- dicom2nifti is promising
    # Put extra metadata in sidecar JSON file
    #
    # NOTE: This is not the final version of this method.
    # The conversion from DICOM to BIDS-I and gathering all required metadata
    # can be complex, as DICOM doesn't necessarily have the metadata required
    # for BIDS in it by default. Thus, another component will handle the logic
    # and error handling surrounding this.
    niftiImage = convertDicomImgToNifti(dicomImg)
    logger.debug("Nifti header after conversion is: %s", niftiImage.header)
    publicMeta, privateMeta = getMetadata(dicomImg)

    publicMeta.update(privateMeta)  # combine metadata dictionaries
    requiredMetadata = {'sub': '002', 'task': 'story', 'contrast_label': 'bold'}
    publicMeta.update(requiredMetadata)
    return BidsIncremental(niftiImage, publicMeta)

def verifyNiftiHeadersMatch(img1: nib.Nifti1Image, img2: nib.Nifti1Image):
    """
    Verifies that two Nifti image headers match in along a defined set of
    NIfTI header fields which should not change during a continuous fMRI
    scanning session.

    This is primarily intended as a safety check, and does not conclusively
    determine that two images are valid to append to together or are part of the
    same scanning session.

    Args:
        header1: First Nifti header to compare (dict of numpy arrays)
        header2: Second Nifti header to compare (dict of numpy arrays)

    Returns:
        True if the headers match along the required dimensions, false
        otherwise.

    """
    fieldsToMatch = ["intent_p1", "intent_p2", "intent_p3", "intent_code",
                     "dim_info", "datatype", "bitpix", "xyzt_units",
                     "slice_duration", "toffset", "scl_slope", "scl_inter",
                     "qform_code", "quatern_b", "quatern_c", "quatern_d",
                     "qoffset_x", "qoffset_y", "qoffset_z",
                     "sform_code", "srow_x", "srow_y", "srow_z"]

    header1 = img1.header
    header2 = img2.header

    for field in fieldsToMatch:
        v1 = header1.get(field)
        v2 = header2.get(field)

        # Use slightly more complicated check to properly match nan values
        if not (np.allclose(v1, v2, atol=0.0, equal_nan=True)):
            logger.debug("Nifti headers don't match on field: %s \
                         (v1: %s, v2: %s)\n", field, v1, v2)
            if DISABLE_NIFTI_HEADER_CHECK:
                continue
            else:
                return False

    # For pixel dimensions, 0 and 1 are equivalent -- any value in a higher
    # index than the number of dimensions specified in the 'dim' field will be
    # ignored, and a 0 in a non-ignored index makes no sense
    field = "pixdim"
    v1 = header1.get(field)
    v2 = header2.get(field)
    v1 = np.where(v1 == 0, 1, v1)
    v2 = np.where(v2 == 0, 1, v2)

    if not (np.allclose(v1, v2, atol=0.0, equal_nan=True)):
        logger.debug("Nifti headers don't match on field: %s \
                     (v1: %s, v2: %s)\n", field, v1, v2)
        if DISABLE_NIFTI_HEADER_CHECK:
            pass
        else:
            return False

    return True

def verifyMetadataMatch(meta1: dict, meta2: dict):
    """
    Verifies two metadata dictionaries match in a set of required fields. If a
    field is present in only one or neither of the two dictionaries, this is
    considered a match.

    This is primarily intended as a safety check, and does not conclusively
    determine that two images are valid to append to together or are part of the
    same series.

    Args:
        meta1: First metadata dictionary
        meta2: Second metadata dictionary

    Returns:
        True if all keys that are present in both dictionaries have equivalent
        values, False otherwise.

    """
    if DISABLE_METADATA_CHECK:
        return True

    matchFields = ["Modality", "MagneticFieldStrength", "ImagingFrequency",
            "Manufacturer", "ManufacturersModelName", "InstitutionName",
            "InstitutionAddress", "DeviceSerialNumber", "StationName",
            "BodyPartExamined", "PatientPosition", "ProcedureStepDescription",
            "SoftwareVersions", "MRAcquisitionType", "SeriesDescription",
            "ProtocolName", "ScanningSequence", "SequenceVariant",
            "ScanOptions", "SequenceName", "ImageType", "SliceThickness",
            "SpacingBetweenSlices", "EchoTime", "RepetitionTime", "FlipAngle",
            "PartialFourier", "ImageOrientationPatientDICOM",
            "InPlanePhaseEncodingDirectionDICOM", "PhaseEncodingDirection"]

    # If either field is None, short-circuit and continue checking other fields
    for field in matchFields:
        field1 = meta1.get(field)
        if field1 == None:
            continue

        field2 = meta2.get(field)
        if field2 == None:
            continue

        if field1 != field2:
            logger.debug("Metadata doen't match on field: %s \
                         (v1: %s, v2: %s)\n", field, field1, field2)
            return False

    # These fields should not match between two images for a valid append
    differentFields = ["AcquisitionTime", "Acquisition Number"]

    for field in differentFields:
        logger.debug("Verifying: %s", field)
        field1 = meta1.get(field)
        if field1 == None:
            continue

        field2 = meta2.get(field)
        if field2 == None:
            continue

        if field1 == field2:
            logger.debug("Metadata matches (shouldn't) on field: %s \
                         (v1: %s, v2: %s)\n", field, field1, field2)
            return False

    return True

def appendBidsinc(incremental: BidsIncremental,
                  archive: BidsArchive,
                  makePath: bool = False) -> None:
    """
    Appends the provided BIDS Incremental imaging and metadata to the provided
    BIDS archive. By default, expects that the incremental represents a valid
    subset of the archive and no additional directory paths will need to be
    created within the archive (this behavior can be overriden).

    Args:
        incremental: BIDS Incremental file containing image data and metadata
        archive: BIDS Archive file to append image data and metadata to
        makePath: Create the directory path for the BIDS-I in the archive if it
            doesn't already exist.

    Returns:
        None

    Raises:
        ValidationError: If the image path within the BIDS incremental does not
            match any existing paths within the archive, and no override is set

    """
    # 1) Create target path for image in archive
    imgDirPath = incremental.makeDataDirPath()
    imgPath = os.path.join(imgDirPath, incremental.getImageFileName())
    metadataPath = os.path.join(imgDirPath, incremental.getMetadataFileName())

    # 2) Verify we have a valid way to append the image to the archive. 3 cases:
    # 2.1) Image already exists within archive, append this Nifti to that Nifti
    # 2.2) Image doesn't exist in archive, but rest of the path is valid for the
    # archive; create new Nifti file within the archive
    # 2.3) Neither image nor path is valid for provided archive; fail append
    if archive.pathExists(imgPath):
        logger.debug("Image exists in archive, appending")
        archiveImg = archive.getImage(imgPath)

        # Validate header match
        if not verifyNiftiHeadersMatch(incremental.image,
                archiveImg):
            raise ValidationError("Nifti headers failed validation!")
        if not verifyMetadataMatch(incremental.imgMetadata,
                archive.getMetadata(metadataPath)):
            raise ValidationError("Image metadata failed validation!")

        # Build 4-D NIfTI if archive has 3-D, concat to 4-D otherwise
        incrementalData = incremental.image.get_fdata()
        archiveData = archiveImg.get_fdata()

        if len(archiveData.shape) == 3:
            newArchiveData = np.stack((archiveData, incrementalData), axis=3)
        else:
            incrementalData = np.expand_dims(incrementalData, 3)
            newArchiveData = np.concatenate((archiveData, incrementalData), axis=3)

        newImg = nib.Nifti1Image(newArchiveData,
                                 archiveImg.affine,
                                 header=archiveImg.header)
        newImg.update_header()
        archive.addImage(newImg, imgPath)

    elif archive.pathExists(imgDirPath) or makePath is True:
        logger.debug("Image doesn't exist in archive, creating")
        archive.addImage(incremental.image, imgPath)
        archive.addMetadata(incremental.imgMetadata, metadataPath)

    else:
        raise ValidationError("No valid archive path for image and no override \
                               specified, can't append")

def bidsToBidsinc(bidsArchive: BidsArchive,
                  subjectID: str,
                  sessionID: str,
                  dataType: str,
                  otherLabels: str,
                  imageIndex: int):
    """
    Returns a Bids-Incremental file with the specified image of the bidsarchive
    Args:
        bidsArchive: The archive to pull data from
        subjectID: Subject ID to pull data for (for "sub-control01", ID is
            "control01")
        sessionID: Session ID to pull data for (for "ses-2020", ID is "2020")
        dataType: Type of data to pull (common types: anat, func, dwi, fmap).
            This string must be the same as the name of the directory containing
            the image data.
        otherLabels: Other labels specifying appropriate file to pull data for
            (for "sub-control01_task-nback_bold", label is "task-nback_bold").
        imageIndex: Index of image to pull if specifying a 4-D sequence.
    Examples:
        bidsToBidsInc(archive, "01", "2020", "func", "task-nback_bold", 0) will
        extract the first image of the volume at:
        "sub-01/ses-2020/func/sub-01_task-nback_bold.nii"
    """
    pass
