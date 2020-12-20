"""-----------------------------------------------------------------------------

bidsLibrary.py

Implement conversion between DICOM, BIDS-Incremental (BIDS-I), and BIDS.

-----------------------------------------------------------------------------"""
import json
import logging
import os

from bids.layout import parse_file_entities
import nibabel as nib
import numpy as np
import pydicom

from rtCommon.bidsArchive import BidsArchive
from rtCommon.bidsIncremental import BidsIncremental
import rtCommon.bidsCommon as bidsCommon
from rtCommon.errors import ValidationError
from rtCommon.imageHandling import convertDicomImgToNifti, readNifti

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


def bidsToBidsinc(archive: BidsArchive,
                  subject: str,
                  session: str,
                  task: str,
                  suffix: str,
                  dataType: str,
                  imageIndex: int = 0,
                  otherLabels: dict = None):
    """
    Creates a BIDS-Incremental file from the specified part of the BIDS Archive.

    Args:
        archive: The archive to pull data from
        subject: Subject ID to pull data for (for "sub-control01", ID is
            "control01")
        session: Session ID to pull data for (for "ses-2020", ID is "2020")
        task: Task to pull data for (for "task-nback", name is "nback")
        suffix: BIDS suffix for file, which is image contrast for fMRI (bold,
            cbv, or phase)
        imageIndex: Index of 3_D image to select in a 4-D sequence of images.
        dataType: Type of data to pull (common types: anat, func, dwi, fmap).
            This string must be the same as the name of the directory containing
            the image data.
        otherLabels: Other entity labels specifying appropriate file to pull
            data for (e.g., 'run', 'rec', 'dir', 'echo')

    Returns:
        BIDS-Incremental file with the specified image of the archive and its
            associated metadata

    Examples:
        bidsToBidsInc(archive, "01", "2020", "func", "task-nback_bold", 0) will
        extract the first image of the volume at:
        "sub-01/ses-2020/func/sub-01_task-nback_bold.nii"

    """
    archivePath = "sub-{}".format(subject)
    if session is not None:
        archivePath = os.path.join(archivePath, "ses-{}".format(session))
    archivePath = os.path.join(archivePath, dataType)

    matchingFilePaths = archive.getFilesForPath(archivePath)
    niftiPaths = [path for path in matchingFilePaths
                  if bidsCommon.isNiftiPath(path)]
    metaPaths = [path for path in matchingFilePaths
                 if bidsCommon.isJsonPath(path)]

    # Fail if no images
    if not niftiPaths:
        logger.error("Failed to find any matching images in the archive to "
                     " make a BIDS-I from")
        return None

    # Warn if no metadata
    if not metaPaths:
        logger.warning("Failed to find any matching metadata in the archive to "
                       "include in a BIDS-I")

    metadata = {'sub': subject, 'ses': session, 'task': task,
                'suffix': suffix}
    image = None

    def pathEntitiesMatch(path) -> bool:
        """
        Return true if the BIDS entities contained in the file at the given path
        match the entities provided to the BIDS -> BIDS-I conversion method.
        """
        entities = parse_file_entities(path)

        if entities.get("task") != task or \
           entities.get("subject") != subject or \
           entities.get("session") != session:
            return False

        if otherLabels:
            for label, value in otherLabels.items():
                if entities.get(label) != value:
                    return False

        if suffix and entities.get("suffix") != suffix:
            return False

        return True

    for path in niftiPaths:
        if pathEntitiesMatch(path):
            image = readNifti(path)
            break

    for path in metaPaths:
        if pathEntitiesMatch(path):
            with open(path, 'r', encoding='utf-8') as metadataFile:
                metadata.update(json.load(metadataFile))
            break

    if image is None:
        logger.error("Failed to find matching image in BIDS Archive \
                     for provided metadata")
        return None
    elif len(image.dataobj.shape) == 3:
        if imageIndex != 0:
            logger.error("Matching image was a 3-D NIfTI; time index %d is too "
                         "high for a 3-D NIfTI (must be 0)", imageIndex)
            return None
        return BidsIncremental(image, subject, task, suffix, metadata)
    else:
        slices = nib.funcs.four_to_three(image)[imageIndex]
        if imageIndex < len(slices):
            newImage = slices[imageIndex]
        else:
            logger.error("Image index %d is too large for NIfTI volume of \
                          length %d", imageIndex, len(slices))
        return BidsIncremental(newImage, subject, task, suffix, metadata)
