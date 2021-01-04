"""-----------------------------------------------------------------------------

bidsIncremental.py

Implements the BIDS Incremental data type used for streaming BIDS data between
different applications.

-----------------------------------------------------------------------------"""
from copy import deepcopy
import json
from operator import eq as opeq
import os
import re
from typing import Any, Callable

from bids.layout.writing import build_path as bids_build_path
import logging
import nibabel as nib
import numpy as np

from rtCommon.errors import ValidationError
from rtCommon.bidsCommon import (
    BidsEntityKeys as bek,
    BidsFileExtension,
    BIDS_FILE_PATTERN,
    BIDS_DIR_PATH_PATTERN,
    DATASET_DESC_REQ_FIELDS,
    DEFAULT_DATASET_DESC,
    getNiftiData,
    loadBidsEntities,
)

logger = logging.getLogger(__name__)


class BidsIncremental:
    # TODO(spolcyn): Rename this to something more useful
    ENTITIES = loadBidsEntities()
    REQUIRED_IMAGE_METADATA = ["subject", "task", "suffix",
                               "RepetitionTime", "EchoTime"]

    """
    BIDS Incremental data format suitable for streaming BIDS Archives
    """
    def __init__(self,
                 image: nib.Nifti1Image,
                 imageMetadata: dict,
                 datasetMetadata: dict = None):
        """
        Initializes a BIDS Incremental object with provided image and metadata.

        Args:
            image: Nifti image as an NiBabel NiftiImage.
            imgMetadata: Metadata for image
            datasetMetadata: Top-level dataset metadata for the BIDS dataset
                to be placed in a dataset_description.json.

        Raises:
            ValidationError: If any required argument is None.

        """

        """ Do basic input validation """
        # IMAGE
        if image is None or \
                (type(image) is not nib.Nifti1Image and
                 type(image) is not nib.Nifti2Image):
            raise ValidationError("Image must be NIBabel Nifti 1 or 2 image, "
                                  "got type %s" % str(type(image)))
        # IMAGE METADATA
        missingImageMetadata = self.missingImageMetadata(imageMetadata)
        if missingImageMetadata != []:
            raise ValidationError(f"Image metadata missing required fields: "
                                  f"{missingImageMetadata}")

        # DATASET METADATA
        if datasetMetadata is not None:
            missingFields = [field for field in DATASET_DESC_REQ_FIELDS
                             if datasetMetadata.get(field, None) is None]
            if missingFields:
                raise ValidationError(
                    f"Dataset description needs: {str(missingFields)}")

        """ Store image metadata """
        # Ensure BIDS-I has an independent metadata dictionary
        # TODO(spolcyn): Replace this deepcopy with building a dictionary of
        # BIDSEntity types from PyBids
        self._imgMetadata = deepcopy(imageMetadata)

        protocolName = self._imgMetadata.get("ProtocolName", None)
        self._imgMetadata.update(self.metadataFromProtocolName(protocolName))

        # Validate or modify fields that are now known to exist
        # TODO(spolcyn): Make a more extensible approach using PyBids entities
        # to putting values into the proper types
        if self._imgMetadata.get("run", None):
            self._imgMetadata["run"] = int(self._imgMetadata["run"])

        self._imgMetadata["TaskName"] = self._imgMetadata["task"]

        # Ensure datatype is set, assume functional
        if self._imgMetadata.get("datatype", None) is None:
            self._imgMetadata["datatype"] = "func"

        fieldToMaxValue = {"RepetitionTime": 100, "EchoTime": 1}
        for field, maxValue in fieldToMaxValue.items():
            value = int(self.getMetadataField(field))
            if value <= maxValue:
                continue
            elif value / 1000.0 <= maxValue:
                logger.info(f"{field} has value {value} > {maxValue}. Assuming "
                            f"value is in milliseconds, converting to seconds.")
                value = value / 1000.0
                self.setMetadataField(field, value)
            else:
                raise ValidationError(f"{field} has value {value}, which is "
                                      f"greater than {maxValue} even if "
                                      f"interpreted as milliseconds.")

        """ Store dataset metadata """
        if datasetMetadata is None:
            self.datasetMetadata = DEFAULT_DATASET_DESC
        else:
            self.datasetMetadata = deepcopy(datasetMetadata)

        """ Validate and store image """
        # Remove singleton dimensions
        self.image = nib.funcs.squeeze_image(image)

        # Validate dimensions, upgrading if needed
        imageShape = self.imageDimensions()
        if len(imageShape) < 3:
            raise ValidationError("Image must have at least 3 dimensions")
        elif len(imageShape) == 3:
            # Add one singleton dimension to make image 4-D
            newData = np.expand_dims(self.image.dataobj, -1)
            self.image = self.image.__class__(newData, self.image.affine,
                                              self.image.header)
            # Update the time dimension size with the TR length
            self.imageHeader["pixdim"][4] = \
                self.getMetadataField("RepetitionTime")

        self._imageDataArray = getNiftiData(self.image)
        assert len(self.imageDimensions()) == 4

        # Configure additional required BIDS metadata and files
        self.readme = "Generated BIDS-Incremental Dataset from RT-Cloud"

        # The BIDS-I version for serialization
        self.version = 1

    def __str__(self):
        return "Image shape: {}; # Metadata Keys: {}; Version: {}".format(
            self.imageDimensions(),
            len(self._imgMetadata.keys()),
            self.version)

    def __eq__(self, other):
        def symmetricDifference(d1: dict, d2: dict,
                                equal: Callable[[Any, Any], bool],
                                ) -> dict:
            sharedKeys = d1.keys() & d2.keys()
            difference = {key: [d1[key], d2[key]]
                          for key in sharedKeys
                          if not equal(d1[key], d2[key])}

            d1OnlyKeys = d1.keys() - d2.keys()
            difference.update({key: [d1[key], None] for key in d1OnlyKeys})

            d2OnlyKeys = d2.keys() - d1.keys()
            difference.update({key: [None, d2[key]] for key in d2OnlyKeys})

            return difference

        def reportDifference(valueName: str, d1: dict, d2: dict,
                             equal: Callable[[Any, Any], bool] = opeq) -> None:
            logger.debug(valueName + " didn't match")
            difference = symmetricDifference(d1, d2, equal)
            logger.debug(valueName + " difference: %s", difference)

        # Compare images
        if self.image.header != other.image.header:
            reportDifference("Image headers",
                             dict(self.image.header),
                             dict(other.image.header),
                             np.array_equal)
            return False

        if self.imageDimensions() != other.imageDimensions():
            logger.debug("Image dimensions didn't match")
            logger.debug("Dimension 1: %s | Dimension 2: %s",
                         self.imageDimensions(), other.imageDimensions())
            return False

        # Compare image metadata
        if self._imgMetadata != other._imgMetadata:
            reportDifference("Image metadata",
                             self._imgMetadata,
                             other._imgMetadata,
                             np.array_equal)
            return False

        # Compare dataset metadata
        if self.datasetMetadata != other.datasetMetadata:
            reportDifference("Dataset metadata",
                             self.datasetMetadata,
                             other.datasetMetadata)
            return False

        if not np.array_equal(self.imageData(), other.imageData()):
            differences = self.imageData() != other.imageData()
            logger.debug("Image data didn't match")
            logger.debug("Difference count: %d (%f%%)",
                         np.sum(differences),
                         np.sum(differences) / np.size(differences) * 100.0)
            return False

        return True

    @staticmethod
    def createImageMetadataDict(subject: str, task: str, suffix: str,
                                repetitionTime: int, echoTime: int):
        """
        Creates an image metadata dictionary for a BIDS-I with all of the
        basic required fields using the correct key names.

        Args:
            subject: Subject ID (e.g., '01')
            task: Task ID (e.g., 'story')
            suffix: Data type (e.g., 'bold')
            repetitionTime: TR time, in seconds, used for the imaging run
            echoTime: Echo time, in seconds, used for the imaging run

        Returns:
            Dictionary with the provided information ready for use in a BIDS-I

        """
        return {"subject": subject, "task": task, "suffix": suffix,
                "RepetitionTime": repetitionTime, "EchoTime": echoTime}

    @classmethod
    def missingImageMetadata(cls, imageMeta: dict) -> list:
        return [f for f in cls.REQUIRED_IMAGE_METADATA if f not in imageMeta]

    @classmethod
    def isCompleteImageMetadata(cls, imageMeta: dict) -> bool:
        """
        Verifies that all required metadata fields for BIDS-I construction are
        present in the dictionary.

        Args:
            imageMeta: The dictionary with the metadata fields

        Returns:
            True if all required fields are present in the dictionary, False
            otherwise.

        """
        return len(cls.missingImageMetadata(imageMeta)) == 0

    @classmethod
    def metadataFromProtocolName(cls, protocolName: str) -> dict:
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
        for entityName, entityValueDict in cls.ENTITIES.items():
            entity = entityValueDict[bek.ENTITY.value]
            entitySearchPattern = patternTemplate.format(field=entity)
            result = re.search(entitySearchPattern, protocolName)

            if result is not None and len(result.groups()) == 1:
                foundEntities[entityName] = result.group(1)

        return foundEntities

    def _exceptIfNotBids(self, entityName: str):
        """ Raise an exception if the argument is not a valid BIDS entity """
        if self.ENTITIES.get(entityName) is None:
            raise ValueError(f"{entityName} is not a valid BIDS entity name")

    # TODO(spolcyn): Add specific getters for commonly used things, like getRun,
    # getSubject, getTask
    def getMetadataField(self, field: str, strict: bool = False) -> str:
        """
        Retrieve value for the metadata field, if it exists.

        Args:
            field: Metadata field to retrieve a value for.
            strict: Only allow getting of official BIDS entity fields.

        Returns:
            Entity's value, or None if the entity isn't present in the metadata

        Raises:
            ValueError if 'strict' is True and 'field' is not a BIDS entity.
        """
        if strict:
            self._exceptIfNotBids(field)
        return self._imgMetadata.get(field, None)

    def setMetadataField(self, field: str, value, strict: bool = False) -> None:
        """
        Set metadata field to provided value.

        Args:
            field: Metadata field to set value for.
            value: Value to set for the provided entity.
            strict: Only allow setting of official BIDS entity fields.

        Raises:
            ValueError if 'strict' is True and 'field' is not a BIDS entity.
        """
        if strict:
            self._exceptIfNotBids(field)
        if field:
            self._imgMetadata[field] = value
        else:
            raise ValueError("Metadata field to set cannot be None")

    def removeMetadataField(self, field: str, strict: bool = False) -> None:
        """
        Remove a piece of metadata.

        Args:
            field: BIDS entity name to retrieve a value for.
            strict: Only allow remove of official BIDS entity fields.

        Raises:
            ValueError if 'strict' is True and 'field' is not a BIDS entity.
        """
        if field in self.REQUIRED_IMAGE_METADATA:
            raise ValueError(f"\"{field}\" is required and cannot be removed")

        if strict:
            self._exceptIfNotBids(field)
        self._imgMetadata.pop(field, None)

    def suffix(self) -> str:
        return self._imgMetadata.get("suffix")

    # Additional methods to access internal BIDS-I data
    def dataType(self):
        """ func or anat """
        return self._imgMetadata.get("datatype")

    @property
    def imgMetadata(self):
        return self._imgMetadata.copy()

    # Getting internal NIfTI data
    def imageData(self) -> np.ndarray:
        return self._imageDataArray

    @property
    def imageHeader(self):
        return self.image.header

    def imageDimensions(self) -> tuple:
        return self.imageHeader.get_data_shape()

    """
    BEGIN BIDS-I ARCHIVE EMULTATION API

    A BIDS-I is meant to emulate a valid BIDS archive. Thus, an API is included
    that enables generating paths and filenames that would corresopnd to this
    BIDS-I's data if it were actually in an on-disk archive.

    """
    def makeBidsFileName(self, extension: BidsFileExtension) -> str:
        """
        Create the a BIDS-compatible file name based on the metadata. General
        format of the filename, per BIDS standard 1.4.1, is as follows (items in
        [square brackets] are considered optional):

        sub-<label>[_ses-<label>]_task-<label>[_acq-<label>] [_ce-<label>]
        [_dir-<label>][_rec-<label>][_run-<index>]
        [_echo-<index>]_<contrast_label >.ext

        Args:
            extension: The extension for the file, e.g., 'nii' for images or
                'json' for metadata

        Return:
            Filename from metadata according to BIDS standard 1.4.1.
        """
        entities = {key: self._imgMetadata[key] for key in self.ENTITIES.keys()
                    if self._imgMetadata.get(key, None) is not None}

        entities["extension"] = extension.value
        if extension == BidsFileExtension.EVENTS:
            entities["suffix"] = "events"
        else:
            entities["suffix"] = self.imgMetadata["suffix"]

        return bids_build_path(entities, BIDS_FILE_PATTERN)

    def imageFileName(self) -> str:
        return self.makeBidsFileName(BidsFileExtension.IMAGE)

    def metadataFileName(self) -> str:
        return self.makeBidsFileName(BidsFileExtension.METADATA)

    def eventsFileName(self) -> str:
        return self.makeBidsFileName(BidsFileExtension.EVENTS)

    def datasetName(self) -> str:
        return self.datasetMetadata["Name"]

    def imageFilePath(self) -> str:
        return os.path.join(self.dataDirPath(), self.imageFileName())

    def metadataFilePath(self) -> str:
        return os.path.join(self.dataDirPath(), self.metadataFileName())

    def dataDirPath(self) -> str:
        """
        Path to where this incremental's data would be in a BIDS archive,
        relative to the archive root.

        Returns:
            Path string relative to root of the imaginary dataset.
        Examples:
            >>> print(bidsi.dataDirPath())
            /sub-01/ses-2011/anat/

        """
        path = bids_build_path(self._imgMetadata, BIDS_DIR_PATH_PATTERN)
        return f"/{path}/"

    def writeToArchive(self, datasetRoot: str):
        """
        Writes the incremental's data to a directory on disk. NOTE: The
        directory is assumed to be empty, and no checks are made for data that
        would be overwritten.

        Args:
            datasetRoot: Path to the root of the BIDS archive to be written to.
        """
        # Build path to the data directory
        # Overall hierarchy:
        # sub-<participant_label>/[/ses-<session_label>]/<data_type>/
        pathElements = ['sub-' + self.getMetadataField("subject")]

        session = self.getMetadataField("session")
        if session:
            pathElements.append('ses-' + session)

        pathElements.append(self.dataType())

        dataDirPath = os.path.join(datasetRoot, *pathElements)
        os.makedirs(dataDirPath, exist_ok=True)

        # Write image to data folder
        imagePath = os.path.join(dataDirPath, self.imageFileName())
        nib.save(self.image, imagePath)

        # Write out image metadata
        metadataPath = os.path.join(dataDirPath, self.metadataFileName())
        with open(metadataPath, mode='w') as metadataFile:
            # TODO(spolcyn): Write out internally used entities, like "subject",
            # "task", and "run" in a suitable way (likely not at all, since
            # they're included in the filename)
            # TODO(spolcyn): Remove after done testing for full bids validation
            # TODO(spolcyn): Support volume timing and associated fields
            self._imgMetadata["SliceTiming"] = list(np.linspace(0.0, 1.5, 27))

            json.dump(self._imgMetadata, metadataFile, sort_keys=True, indent=4)

        # TODO(spolcyn): Make events file correspond correctly to the imaging
        # sequence, not just to fulfill BIDS validation
        eventsPath = os.path.join(dataDirPath, self.eventsFileName())
        with open(eventsPath, mode='w') as eventsFile:
            eventsFile.write('onset\tduration\tresponse_time\n')

        # Write out dataset description
        descriptionPath = os.path.join(datasetRoot, "dataset_description.json")
        with open(descriptionPath, mode='w') as description:
            json.dump(self.datasetMetadata, description, indent=4)

        # Write out readme
        with open(os.path.join(datasetRoot, "README"), mode='w') as readme:
            readme.write(self.readme)

    """ END BIDS-I ARCHIVE EMULTATION API """
