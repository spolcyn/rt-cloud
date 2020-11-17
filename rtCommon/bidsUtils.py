"""-----------------------------------------------------------------------------

bidsUtils.py

Utilitiy functions for working with BIDS archives.

-----------------------------------------------------------------------------"""

from functools import lru_cache
import os
import pydicom
import re

from rtCommon.errors import ValidationError


def isNiftiPath(path: str) -> bool:
    """
    Returns true if the provided path points to an uncompressed NIfTI file,
    false otherwise.
    """
    _, ext = os.path.splitext(path)
    return ext == '.nii'


@lru_cache(maxsize=128)
def makeDicomFieldBidsCompatible(field: str) -> str:
    """
    Remove characters to make field a BIDS-compatible
    (CamelCase alphanumeric) metadata field.

    NOTE: Keys like 'Frame of Reference UID' become 'FrameofReferenceUID', which
    might be different than the expected behavior
    """
    return re.compile('[^a-zA-z]').sub("", field)


@lru_cache(maxsize=128)
def dicomMetadataNameToBidsMetadataName(tag) -> str:
    try:
        (_, _, name, _, _) = pydicom.datadict.get_entry(tag)
        return makeDicomFieldBidsCompatible(name)
    except KeyError:
        raise ValidationError("Tag {} not a valid DICOM tag".format(tag))