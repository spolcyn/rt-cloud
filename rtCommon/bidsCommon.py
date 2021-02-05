"""-----------------------------------------------------------------------------

bidsCommon.py

Shared constants and functions used by modules working with BIDS data.

-----------------------------------------------------------------------------"""
from enum import Enum
import functools
import logging
import os
import re
from typing import Any, Callable, Tuple

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

        logging.info("Loaded entities: %s", loadedEntities)
        return loadedEntities


def filterEntities(metadata: dict) -> dict:
    """
    Returns a new dictionary containing all the elements of the argument that
    are valid BIDS entities.
    """
    entities = loadBidsEntities()
    return {key: metadata[key] for key in metadata if key in entities}


def getNiftiData(image: nib.Nifti1Image) -> np.ndarray:
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


# From official nifti1.h
UNIT_TO_CODE = {'unknown': 0, 'meter': 1, 'mm': 2, 'micron': 3, 'sec': 8,
                'msec': 16, 'usec': 24, 'hz': 32, 'ppm': 40, 'rads': 48}
CODE_TO_UNIT = {UNIT_TO_CODE[key]: key for key in UNIT_TO_CODE.keys()}


def correct3DHeaderTo4D(image: nib.Nifti1Image, repetitionTime: int,
                        timeUnitCode: int = 8) -> None:
    """
    Makes necessary changes to the NIfTI header to reflect the increase in
    the datasize from 3D to 4D

    Args:
        image: NIfTI image to modify header for
        repetitionTime: Repetition time for the scan that produced the image
        timeUnitCode: The temporal dimension NIfTI unit code (e.g., millimeters
            is 2, seconds is 8)
    """
    oldShape = image.header.get_data_shape()
    dimensions = len(oldShape)
    if dimensions < 3 or dimensions > 4:
        raise ValueError(f'Image must be 3-D or 4-D (got {dimensions}-D)')

    # dim - only update if currently 3D (possible given 4D image whose data has
    # been changed, and dim is correct but pixdim/xyzt_units need correcting)
    if dimensions == 3:
        newShape = (*oldShape, 1)
        logger.debug(f"Shape old: {oldShape} | Shape new: {newShape}")
        image.header.set_data_shape(newShape)

    # pixdim
    oldZooms = image.header.get_zooms()
    if len(oldZooms) == 3:
        newZooms = (*oldZooms, repetitionTime)
    elif len(oldZooms) == 4:
        newZooms = (*oldZooms[0:3], repetitionTime)
    logger.debug(f"Zooms old: {oldZooms} | Zooms new: {newZooms}")
    image.header.set_zooms(newZooms)

    # xyzt_units
    oldUnits = image.header.get_xyzt_units()
    newUnits = (oldUnits[0], timeUnitCode)
    image.header.set_xyzt_units(xyz=newUnits[0], t=newUnits[1])
    logger.debug(f"Units old: {oldUnits} | Units new: {newUnits}")


def addSecondsToXyztUnits(image: nib.Nifti1Image) -> None:
    oldUnits = image.header.get_xyzt_units()
    image.header.set_xyzt_units(xyz=oldUnits[0], t=8)
    logger.debug("Units before adding seconds: %s", oldUnits)


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
            raise ValueError(f"{field}'s max value is {maxValue}; {value} > "
                             f"{maxValue} even if interpreted as milliseconds.")


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


def getDicomMetadata(dicomImg: pydicom.dataset.Dataset) -> Tuple[dict, dict]:
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


def symmetricDictDifference(d1: dict, d2: dict,
                            equal: Callable[[Any, Any], bool]) -> dict:
    sharedKeys = d1.keys() & d2.keys()
    difference = {key: [d1[key], d2[key]]
                  for key in sharedKeys
                  if not equal(d1[key], d2[key])}

    d1OnlyKeys = d1.keys() - d2.keys()
    difference.update({key: [d1[key], None] for key in d1OnlyKeys})

    d2OnlyKeys = d2.keys() - d1.keys()
    difference.update({key: [None, d2[key]] for key in d2OnlyKeys})

    return difference
