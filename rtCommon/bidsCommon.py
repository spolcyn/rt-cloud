"""-----------------------------------------------------------------------------

bidsCommon.py

Shared constants and functions used by modules working with BIDS data.

-----------------------------------------------------------------------------"""
from enum import Enum
import functools
import os
import re

import logging
import pydicom
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
