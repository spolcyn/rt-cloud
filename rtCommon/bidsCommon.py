"""-----------------------------------------------------------------------------

bidsCommon.py

Shared constants and functions used by modules working with BIDS data.

-----------------------------------------------------------------------------"""
from enum import Enum
import functools
import os
import re

import pydicom
import yaml

from rtCommon.errors import ValidationError

# Version of the standard to be compatible with
BIDS_VERSION = "1.4.1"

# Required fields in the dataset_description.json file
DATASET_DESC_REQ_FIELDS = ["Name", "BIDSVersion"]


# Valid extensions for various file types in the BIDS format
class BidsFileExtension(Enum):
    IMAGE = '.nii'
    IMAGE_COMPRESSED = '.nii.gz'
    METADATA = '.json'


# BIDS Entitiy information dict
class BidsEntityKeys(Enum):
    ENTITY_KEY = "entity"
    FORMAT_KEY = "format"
    DESCRIPTION_KEY = "description"


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
