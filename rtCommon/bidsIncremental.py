"""-----------------------------------------------------------------------------

bidsIncremental.py

Implements the BIDS Incremental data type used for streaming BIDS data between
different applications.

-----------------------------------------------------------------------------"""
from copy import deepcopy
import json
import os
import re

import logging
import nibabel as nib
import numpy as np

from rtCommon.errors import ValidationError
from rtCommon.bidsCommon import (
    loadBidsEntities,
    BidsEntityKeys as bek,
    BidsFileExtension,
    BIDS_VERSION,
    DATASET_DESC_REQ_FIELDS,
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
            image: Nifti image for this BIDS-I as an NiBabel NiftiImage.
                NIfTI 2 is not officially tested, but should work without issue.
            subject: Subject ID corresponding to this image
            task: Task name corresponding to this image
            suffix: Image contrast type for this image ('bold' or 'cbv')
            imgMetadata: Additional metadata about the image
            datasetMetadata: Top-level dataset metadata for the BIDS dataset
                this represents (dataset_description.json).

        Raises:
            ValidationError: If any required argument is None.

        """

        """ Validate and store image """
        if  image is None or \
            (not isinstance(image, nib.Nifti1Image) and
             not isinstance(image, nib.Nifti2Image)):
            raise ValidationError("Image must be NIBabel Nifti 1 or 2 image, "
                                  "got type %s" % str(type(image)))

        # Remove singleton dimensions
        self.image = nib.funcs.squeeze_image(image)

        # Validate dimensions
        imageShape = image.get_fdata().shape
        logger.debug("IMage shape: %s", str(imageShape))
        if len(imageShape) < 3:
            raise ValidationError("Image must have at least 3 dimensions")
        elif len(imageShape) == 3:
            # Add one singleton dimension to make image 4-D
            newData = np.expand_dims(image.get_fdata(), -1)

        """ Validate and store image metadata """
        # Ensure BIDS-I has an independent metadata dictionary
        self._imgMetadata = deepcopy(imageMetadata)

        protocolName = self._imgMetadata.get("ProtocolName", None)
        if protocolName is not None:
            self._imgMetadata.update(self.metadataFromProtocolName(protocolName))

        missingImageMetadata = self.missingImageMetadataFields(self._imgMetadata)
        if missingImageMetadata != []:
            raise ValidationError("Image metadata missing required fields: %s" %
                                   str(missingImageMetadata))

        # Validate or modify fields that are now known to exist
        self._imgMetadata["TaskName"] = self._imgMetadata["task"]
        fieldToMaxValue = {"RepetitionTime": 100, "EchoTime": 1}
        for field, maxValue in fieldToMaxValue.items():
            value = int(self._imgMetadata[field])
            if value <= maxValue:
                continue
            elif value / 1000.0 <= maxValue:
                logger.info(f"{field} has value {value} > {maxValue}. Assuming "
                             "value is in milliseconds, converting to seconds.")
                value = value / 1000.0
                self._imgMetadata[field] = value
            else:
                raise ValidationError(f"{field} has value {value}, which is "
                                      f"greater than {maxValue} even if "
                                       "interpreted as milliseconds.")

        """ Validate dataset metadata or create default values """
        if datasetMetadata is not None:
            missingFields = [field for
                             field in DATASET_DESC_REQ_FIELDS
                             if datasetMetadata.get(field) is None]

            if missingFields == []:
                self.datasetMetadata = deepcopy(datasetMetadata)
            else:
                raise ValidationError("Dataset description missing these "
                                      "required fields: " + str(missingFields))
        else:
            self.datasetMetadata = {"Name": "bidsi_dataset",
                                    "BIDSVersion": str(BIDS_VERSION),
                                    "Authors": ["The RT-Cloud Authors",
                                                "The Dataset Author"]}

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
        # Compare images
        # TODO(spolcyn): Could do this more completely by save()'ing the images
        # to disk, then diffing the files that they produce; however, this could
        # be quite slow
        if self.image.header != other.image.header:
            return False
        if not np.array_equal(self.image.get_fdata(), other.image.get_fdata()):
            return False

        # Compare image metadata
        if not self._imgMetadata == other._imgMetadata:
            return False

        # Compare dataset metadata
        if not self.datasetMetadata == other.datasetMetadata:
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
    def missingImageMetadataFields(cls, imageMeta: dict) -> list:
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
        return len(cls.missingImageMetadataFields(imageMeta)) == 0

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
            errorMsg = "'{}' is not a valid BIDS entity name".format(entityName)
            logger.debug(errorMsg)
            raise ValueError(errorMsg)

    # TODO(spolcyn): Add specific getters for commonly used things, like getRun,
    # getSubject, getTask
    # From the BIDS Spec: "A file name consists of a chain of entities, or
    # key-value pairs, a suffix and an extension."
    # Thus, we provide a set of methods to extract these values from the BIDS-I.
    def getMetadataField(self, field: str, strict: bool=False) -> str:
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

    def setMetadataField(self, field: str, value, strict: bool=False) -> None:
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

    def removeMetadataField(self, field: str, strict: bool=False) -> None:
        """
        Remove a piece of metadata.

        Args:
            field: BIDS entity name to retrieve a value for.
            strict: Only allow remove of official BIDS entity fields.

        Raises:
            ValueError if 'strict' is True and 'field' is not a BIDS entity.
        """
        if field in self.REQUIRED_IMAGE_METADATA:
            raise ValueError(f"\"{field}\" is required and cannot be "
                              "removed from metadata.")

        if strict:
            self._exceptIfNotBids(field)
        self._imgMetadata.pop(field, None)

    def suffix(self) -> str:
        return self._imgMetadata.get("suffix")

    # Additional methods to access internal BIDS-I data
    def dataType(self):
        # TODO(spolcyn): Support anatomical imaging too
        """ func or anat """
        return "func"

    # Getting internal NIfTI data
    def imageData(self):
        return self.image.get_fdata()

    def imageHeader(self):
        return self.image.header

    def imageDimensions(self) -> np.ndarray:
        return self.image.header.get("dim")

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
        # (entity, required in path) tuples to specify the order and necessity
        # of how entities appear in the pathname
        entitiesToAdd = ['subject', 'session', 'task', 'acquisition',
                         'contrast enhancing agent', 'phase-encoding direction',
                         'reconstruction', 'run', 'echo']

        entityPairs = []
        for entity in entitiesToAdd:
            value = self.getMetadataField(entity)
            if value:
                shortName = self.ENTITIES.get(entity).get(bek.ENTITY.value)
                entityPairs.append(shortName + '-' + value)

        # Add contrast label
        if extension == BidsFileExtension.EVENTS:
            entityPairs.append("events")
        else:
            entityPairs.append(self.suffix())

        return '_'.join(entityPairs) + extension.value

    def imageFileName(self) -> str:
        return self.makeBidsFileName(BidsFileExtension.IMAGE)

    def metadataFileName(self) -> str:
        return self.makeBidsFileName(BidsFileExtension.METADATA)

    def eventsFileName(self) -> str:
        return self.makeBidsFileName(BidsFileExtension.EVENTS)

    def datasetName(self) -> str:
        return self.datasetMetadata["Name"]

    def imageFilePath(self) -> str:
        return os.path.join(self.makeDataDirPath(), self.imageFileName())

    def dataDirPath(self) -> str:
        """
        Returns the path to the data directory this incremental's data would be
        in if it were in a full BIDS archive.

        Returns:
            Path string relative to root of the imaginary dataset.
        Examples:
            >>> print(bidsi.dataDirPath())
            /sub-01/ses-2011/anat/

        """
        pathElements = ['sub-' + self.getMetadataField("subject")]

        session = self.getMetadataField("session")
        if session:
            pathElements.append('ses-' + session)

        pathElements.append(self.dataType())
        pathElements.append("")

        return os.path.join("/", *pathElements)

    # TODO(spolcyn): Write a BIDS-I to a valid BIDS Archive on disk
    # TODO(spolcyn): Call BIDS Validator on that BIDS Archive
    def writeToArchive(self, directoryPath: str):
        """
        Args:
            directoryPath: Location to write the BIDS-I derived BIDS Archive
        """
        # Build path to the data directory
        # Overall hierarchy:
        # sub-<participant_label>/[/ses-<session_label>]/<data_type>/
        pathElements = ['sub-' + self.getMetadataField("subject")]

        session = self.getMetadataField("session")
        if session:
            pathElements.append('ses-' + session)

        pathElements.append(self.dataType())

        datasetDir = os.path.join(directoryPath, self.datasetName())
        dataDirPath = os.path.join(datasetDir, *pathElements)
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
        descriptionPath = os.path.join(datasetDir, "dataset_description.json")
        with open(descriptionPath, mode='w') as description:
            json.dump(self.datasetMetadata, description)

        # Write out readme
        with open(os.path.join(datasetDir, "README"), mode='w') as readme:
            readme.write(self.readme)

    """ END BIDS-I ARCHIVE EMULTATION API """
