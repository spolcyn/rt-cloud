"""-----------------------------------------------------------------------------

bidsLibrary.py

Implement conversion between DICOM, BIDS-Incremental (BIDS-I), and BIDS.

-----------------------------------------------------------------------------"""
import logging

import nibabel as nib
import numpy as np
import pydicom

from rtCommon.bidsIncremental import BidsIncremental
import rtCommon.bidsCommon as bidsCommon
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

        cleanedKey = bidsCommon.makeDicomFieldBidsCompatible(elem.name)
        # in DICOM, public tags have even group numbers and private tags are odd
        # http://dicom.nema.org/dicom/2013/output/chtml/part05/chapter_7.html
        value = str(elem.value)

        if elem.tag.is_private:
            privateMeta[cleanedKey] = value
        else:
            publicMeta[cleanedKey] = value

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
    requiredMetadata = {'sub': '002', 'task': 'story', 'suffix': 'bold'}
    publicMeta.update(requiredMetadata)
    return BidsIncremental(niftiImage, '002', 'story', 'bold', publicMeta)


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
                   "BodyPartExamined", "PatientPosition", "EchoTime",
                   "ProcedureStepDescription", "SoftwareVersions",
                   "MRAcquisitionType", "SeriesDescription", "ProtocolName",
                   "ScanningSequence", "SequenceVariant", "ScanOptions",
                   "SequenceName", "SpacingBetweenSlices", "SliceThickness",
                   "ImageType", "RepetitionTime", "PhaseEncodingDirection",
                   "FlipAngle", "InPlanePhaseEncodingDirectionDICOM",
                   "ImageOrientationPatientDICOM", "PartialFourier"]

    # If either field is None, short-circuit and continue checking other fields
    for field in matchFields:
        field1 = meta1.get(field)
        if field1 is None:
            continue

        field2 = meta2.get(field)
        if field2 is None:
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
        if field1 is None:
            continue

        field2 = meta2.get(field)
        if field2 is None:
            continue

        if field1 == field2:
            logger.debug("Metadata matches (shouldn't) on field: %s \
                         (v1: %s, v2: %s)\n", field, field1, field2)
            return False

    return True
