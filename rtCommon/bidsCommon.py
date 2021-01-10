"""-----------------------------------------------------------------------------

bidsCommon.py

Shared constants and functions used by modules working with BIDS data.

-----------------------------------------------------------------------------"""
from enum import Enum
import functools
import logging
import os
import re

import pydicom
import nibabel as nib
import numpy as np
import yaml

from rtCommon.errors import ValidationError

logger = logging.getLogger(__name__)

# Version of the standard to be compatible with
BIDS_VERSION = "1.4.1"

# Required fields in the dataset_description.json file
DATASET_DESC_REQ_FIELDS = ["Name", "BIDSVersion"]
DEFAULT_DATASET_DESC = {"Name": "bidsi_dataset",
                        "BIDSVersion": str(BIDS_VERSION),
                        "Authors": ["The RT-Cloud Authors",
                                    "The Dataset Author"]}

# Pattern for creating BIDS filenames from all compatible fMRI entities
BIDS_FILE_PATTERN = "sub-{subject}[_ses-{session}]_task-{task}" \
                     "[_acq-{acquisition}][_ce-{ceagent}][_dir-{direction}]" \
                     "[_rec-{reconstruction}][_run-{run}][_echo-{echo}]" \
                     "[_recording-{recording}][_part-{part}]" \
                     "_{suffix<bold|cbv|sbref|events>}" \
                     "[{extension<.nii|.json|.tsv>}]"

# Pattern for creating BIDS archive directory path
BIDS_DIR_PATH_PATTERN = "sub-{subject}[/ses-{session}]/{datatype<func>|func}"

# Pattern for creating full path of BIDS file relative to archive root
BIDS_FILE_PATH_PATTERN = BIDS_DIR_PATH_PATTERN + '/' + BIDS_FILE_PATTERN


# Valid extensions for various file types in the BIDS format
class BidsFileExtension(Enum):
    IMAGE = '.nii'
    IMAGE_COMPRESSED = '.nii.gz'
    METADATA = '.json'
    EVENTS = '.tsv'


# BIDS Entitiy information dict
class BidsEntityKeys(Enum):
    ENTITY = "entity"
    FORMAT = "format"
    DESCRIPTION = "description"


# See test file for more specifics about expected format
@functools.lru_cache(maxsize=1)
def loadBidsEntities() -> dict:
    # Assumes that this file and entities.yaml are in the same directory
    filename = "entities.yaml"
    rtCommonDir = os.path.dirname(os.path.realpath(__file__))
    filePath = os.path.join(rtCommonDir, filename)

    with open(filePath, mode='r', encoding="utf-8") as entities_file:
        loadedEntities = yaml.safe_load(entities_file)

        entities = {}
        for valueDict in loadedEntities.values():
            name = valueDict["name"]
            del valueDict["name"]
            name = name.lower()
            entities[name] = valueDict

        return entities


def getNiftiData(image) -> np.ndarray:
    """
    Nibabel exposes a get_fdata() method, but this converts all the data to
    float64. Since our Nifti files are often converted from DICOM's, which store
    data in signed or unsigned ints, treating the data as float can cause issues
    when comparing images or re-writing a Nifti read in from disk.
    """
    return np.asanyarray(image.dataobj, dtype=image.dataobj.dtype)


def isNiftiPath(path: str) -> bool:
    """
    Returns true if the provided path points to an uncompressed NIfTI file,
    false otherwise.
    """
    _, ext = os.path.splitext(path)
    return ext == '.nii'


def isJsonPath(path: str) -> bool:
    """
    Returns true if the provided path points to an uncompressed NIfTI file,
    false otherwise.
    """
    _, ext = os.path.splitext(path)
    return ext == '.json'


@functools.lru_cache(maxsize=128)
def makeDicomFieldBidsCompatible(field: str) -> str:
    """
    Remove characters to make field a BIDS-compatible
    (CamelCase alphanumeric) metadata field.

    NOTE: Keys like 'Frame of Reference UID' become 'FrameofReferenceUID', which
    might be different than the expected behavior
    """
    return re.compile('[^a-zA-z]').sub("", field)


@functools.lru_cache(maxsize=128)
def dicomMetadataNameToBidsMetadataName(tag) -> str:
    try:
        (_, _, name, _, _) = pydicom.datadict.get_entry(tag)
        return makeDicomFieldBidsCompatible(name)
    except KeyError:
        raise ValidationError("Tag {} not a valid DICOM tag".format(tag))


def adjustTimeUnits(imageMetadata: dict) -> None:
    """
    Validates and converts in-place the units of various time-based metadata,
    which is stored in seconds in BIDS, but often provided using milliseconds in
    DICOM.
    """
    fieldToMaxValue = {"RepetitionTime": 100, "EchoTime": 1}
    for field, maxValue in fieldToMaxValue.items():
        value = imageMetadata.get(field, None)
        if value is None:
            continue
        else:
            value = int(value)

        if value <= maxValue:
            continue
        elif value / 1000.0 <= maxValue:
            logger.info(f"{field} has value {value} > {maxValue}. Assuming "
                        f"value is in milliseconds, converting to seconds.")
            value = value / 1000.0
            imageMetadata[field] = value
        else:
            raise ValidationError(f"{field}'s max value is {maxValue}; {value} "
                                  f"> {maxValue} even if interpreted as "
                                  f"milliseconds.")


def metadataFromProtocolName(protocolName: str) -> dict:
    """
    Extracts BIDS label-value combinations from a DICOM protocol name, if
    any are present.

    Returns:
        A dictionary containing any valid label-value combinations found.
    """
    if not protocolName:
        return {}

    prefix = "(?:(?<=_)|(?<=^))"  # match beginning of string or underscore
    suffix = "(?:(?=_)|(?=$))"  # match end of string or underscore
    fieldPat = "(?:{field}-)(.+?)"  # TODO(spolcyn): Document this regex
    patternTemplate = prefix + fieldPat + suffix

    foundEntities = {}
    for entityName, entityValueDict in loadBidsEntities().items():
        entity = entityValueDict[BidsEntityKeys.ENTITY.value]
        entitySearchPattern = patternTemplate.format(field=entity)
        result = re.search(entitySearchPattern, protocolName)

        if result is not None and len(result.groups()) == 1:
            foundEntities[entityName] = result.group(1)

    return foundEntities


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

        cleanedKey = makeDicomFieldBidsCompatible(elem.name)
        # in DICOM, public tags have even group numbers and private tags are odd
        # http://dicom.nema.org/dicom/2013/output/chtml/part05/chapter_7.html
        value = str(elem.value)

        if elem.tag.is_private:
            privateMeta[cleanedKey] = value
        else:
            publicMeta[cleanedKey] = value

    return (publicMeta, privateMeta)


# Set to True to skip validating Nifti header when appending
DISABLE_NIFTI_HEADER_CHECK = False

# Set to True to skip validating metadata when appending
DISABLE_METADATA_CHECK = False


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

    # For pixel dimensions, the values 0 and 1 are equivalent -- any value in a
    # higher index than the number of dimensions specified in the 'dim' field
    # will be ignored, and a 0 in a non-ignored index makes no sense
    field = "dim"
    dimensions1 = header1.get(field)[0]
    dimensions2 = header2.get(field)[0]
    nDimensionsToCompare = min(dimensions1, dimensions2)

    field = "pixdim"
    v1 = header1.get(field)[0:nDimensionsToCompare + 1]
    v2 = header2.get(field)[0:nDimensionsToCompare + 1]
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


def verifySidecarMetadataMatch(meta1: dict, meta2: dict):
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
        logger.debug("Analyzing field %s | Meta1: %s | Meta2: %s",
                     field,
                     meta1.get(field),
                     meta2.get(field))
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
