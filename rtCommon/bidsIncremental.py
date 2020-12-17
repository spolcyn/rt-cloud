"""-----------------------------------------------------------------------------

bidsIncremental.py

Implements the BIDS Incremental data type used for streaming BIDS data between
different applications.

-----------------------------------------------------------------------------"""
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
    requiredImageMetadata = ["subject", "task", "suffix",
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
        if image is None:
            raise ValidationError("Image cannot be None")
        elif not (isinstance(image, nib.Nifti1Image) or
                  isinstance(image, nib.Nifti2Image)):
            raise ValidationError("Image must be NIBabel Nifti 1 or 2 image, \
                                   got type %s" % str(type(image)))
        elif not len(image.get_fdata().shape) >= 4:
            # TODO(spolcyn): Make this check datatype specific after adding
            # anatomical data support
            # TODO(spolcyn): Make this check strict
            errorMsg = "BIDS-I only supports 4-D NIfTI volumes for \
                        functional data"
            logger.error(errorMsg)
            # raise ValidationError()

        # Sometimes, image shape has trailing 1's; remove them
        self.image = nib.funcs.squeeze_image(image)
        self.header = image.header

        """ Validate and store image metadata """
        protocolName = imageMetadata.get("ProtocolName", None)
        if protocolName is not None:
            imageMetadata.update(self.metadataFromProtocolName(protocolName))

        missingImageMetadata = self.missingImageMetadataFields(imageMetadata)
        if missingImageMetadata != []:
            raise ValidationError("Image metadata missing required fields: %s" %
                                   str(missingImageMetadata))

        self.imgMetadata = imageMetadata
        self.imgMetadata["TaskName"] = self.imgMetadata["task"]

        # Validate dataset metadata or create default values
        if datasetMetadata is not None:
            missingFields = [field for
                             field in DATASET_DESC_REQ_FIELDS
                             if datasetMetadata.get(field) is None]

            if missingFields != []:
                errorMsg = "Dataset description provided, but missing these \
                        required fields: " + str(missingFields)
                raise ValidationError(errorMsg)
        else:
            datasetMetadata = {}
            datasetMetadata["Name"] = "bidsi_dataset"
            datasetMetadata["BIDSVersion"] = str(BIDS_VERSION)
            if not datasetMetadata.get("Authors"):
                datasetMetadata["Authors"] = ["The RT-Cloud Authors",
                                              "The Dataset Author"]

        self.datasetMetadata = datasetMetadata

        # Configure additional required BIDS metadata and files
        self.readme = "Generated BIDS-Incremental Dataset from RT-Cloud"

        rtFields = ["RepetitionTime", "EchoTime"]
        for field in rtFields:
            value = self.imgMetadata.get(field)
            if not value:
                raise ValidationError("Expected {} field in image metadata"
                                      .format(field))
            elif field == "RepetitionTime":
                value = int(value)
                if value > 100:
                    logger.info("%s %d > 100. Assuming value is in \
                                 milliseconds, converting to seconds.",
                                field,
                                value)
                self.imgMetadata[field] = value / 1000.0
            elif field == "EchoTime":
                value = int(value)
                if value > 1:
                    logger.info("%s %d > 1. Assuming value is in \
                                 milliseconds, converting to seconds.",
                                field,
                                value)
                self.imgMetadata[field] = value / 1000.0

        # The BIDS-I version for serialization
        self.version = 1

    def __str__(self):
        return "Image shape: {}; # Metadata Keys: {}; Version: {}".format(
            self.image.get_fdata().shape,
            len(self.imgMetadata.keys()),
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
        if not self.imgMetadata == other.imgMetadata:
            return False

        # Compare dataset metadata
        if not self.datasetMetadata == other.datasetMetadata:
            return False

        return True

    @classmethod
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
        metadata = {}
        metadata["subject"] = subject
        metadata["task"] = task
        metadata["suffix"] = suffix
        metadata["RepetitionTime"] = repetitionTime
        metadata["EchoTime"] = echoTime

    @classmethod
    def missingImageMetadataFields(cls, imageMeta: dict) -> list:
        return [f for f in cls.requiredImageMetadata if f not in imageMeta]

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
        if protocolName is None:
            return {}

        prefix = "(?:(?<=_)|(?<=^))"  # match beginning of string or underscore
        suffix = "(?:(?=_)|(?=$))"  # match end of string or underscore
        fieldPat = "(?:{field}-)(.+?)"  # TODO(spolcyn): Document this regex
        patternTemplate = prefix + fieldPat + suffix

        foundEntities = {}
        for entityName, entityValueDict in cls.ENTITIES.items():
            entity = entityValueDict[bek.ENTITY_KEY.value]
            entitySearchPattern = patternTemplate.format(field=entity)
            result = re.search(entitySearchPattern, protocolName)

            if result is not None and len(result.groups()) == 1:
                foundEntities[entityName] = result.group(1)

        return foundEntities

    # TODO(spolcyn): Add specific getters for commonly used things, like getRun,
    # getSubject, getTask
    # From the BIDS Spec: "A file name consists of a chain of entities, or
    # key-value pairs, a suffix and an extension."
    # Thus, we provide a set of methods to extract these values from the BIDS-I.
    def getEntity(self, entityName) -> str:
        """
        Retrieve the entity value for the provided full entity name from this
        BIDS Incremental. Be sure to use the full name, e.g., 'subject' instead
        of 'sub'.

        Args:
            entityName: The name of the BIDS entity to retrieve a value for.
                A list of entity names is provided in the BIDS Standard. For
                example, use 'Subject' for subject.

        Returns:
            The entity value, or None if this BIDS incremental doesn't contain a
            value for the entity name.

        """
        if self.ENTITIES.get(entityName) is None:
            errorMsg = "'{}' is not a valid BIDS entity name".format(entityName)
            logger.debug(errorMsg)
            raise ValueError(errorMsg)

        return self.imgMetadata.get(entityName, None)

    def addEntity(self, entityName: str, entityValue: str) -> None:
        """
        Add an entity name-value pair to the BIDS Incremental's metadata. Be
        sure to use the full entity name, e.g., 'subject' instead of 'sub'.

        Args:
            entityName: The name of the BIDS entity to set a value for.
                A list of entity names is provided in the BIDS Standard. For
                example, use 'subject' for subject.
            entityValue: The value to set for the provided entity.

        """
        if self.ENTITIES.get(entityName) is None:
            errorMsg = "'{}' is not a valid BIDS entity name".format(entityName)
            logger.debug(errorMsg)
            raise ValueError(errorMsg)

        self.imgMetadata[entityName] = entityValue

    def removeEntity(self, entityName: str) -> None:
        """
        Remove a entity name-value pair to the BIDS Incremental's metadata. Be
        sure to use the full entity name, e.g., 'subject' instead of 'sub'.

        Args:
            entityName: The name of the BIDS entity to remove.  A list of entity
                names is provided in the BIDS Standard. For example, use
                'subject' for subject.

        """
        if self.ENTITIES.get(entityName) is None:
            errorMsg = "'{}' is not a valid BIDS entity name".format(entityName)
            logger.debug(errorMsg)
            raise ValueError(errorMsg)

        self.imgMetadata.pop(entityName, None)

    def suffix(self) -> str:
        return self.imgMetadata.get("suffix")

    # A BIDS-I produces two files, with different extensions
    def imageExtension(self) -> str:
        return BidsFileExtension.IMAGE

    def metadataExtension(self) -> str:
        return BidsFileExtension.METADATA

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
        return self.image.get("dim")

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
        labelPairs = []  # all potential BIDS field-label pairs in the filename

        labelPairs.append('sub-' + self.getEntity("subject"))

        sesName = self.getEntity("session")
        if sesName:
            labelPairs.append('ses-' + sesName)

        labelPairs.append('task-' + self.getEntity("task"))

        runName = self.getEntity("run")
        if runName:
            labelPairs.append('run-' + runName)

        if extension == BidsFileExtension.EVENTS:
            labelPairs.append("events")
        else:
            labelPairs.append(self.suffix())

        """
        # distinguish using diff params for acquiring same task
        acqLabel = getAcqLabel()
        # "distinguish sequences using different constrast enhanced images"
        ceLabel = getCeLabel()
        # "distinguish different phase-encoding directions"
        dirLabel = getDirLabel()
        # "distinguish different...reconstruction algorithms"
        recLabel = getRecLabel()
        # "more than one run of same task"
        runLabel = getRunLabel()
        # "multi echo data"
        echoLabel = getEchoLabel()
        """

        return '_'.join(labelPairs) + extension.value

    def imageFileName(self) -> str:
        return self.makeBidsFileName(BidsFileExtension.IMAGE)

    def makeImageFilePath(self) -> str:
        dataDir = self.makeDataDirPath()
        return os.path.join(dataDir, self.imageFileName())

    def metadataFileName(self) -> str:
        return self.makeBidsFileName(BidsFileExtension.METADATA)

    def eventsFileName(self) -> str:
        return self.makeBidsFileName(BidsFileExtension.EVENTS)

    def datasetName(self) -> str:
        return self.datasetMetadata["Name"]

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
        pathElements = ['sub-' + self.getEntity("subject")]

        session = self.getEntity("session")
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
        datasetDir = os.path.join(directoryPath, self.datasetName())
        pathElements = ['sub-' + self.getEntity("subject")]

        session = self.getEntity("session")
        if session:
            pathElements.append('ses-' + session)

        pathElements.append(self.dataType())

        dataDirPath = os.path.join(datasetDir, *pathElements)
        os.makedirs(dataDirPath, exist_ok=True)

        # Write image to data folder
        imagePath = os.path.join(dataDirPath, self.imageFileName())
        nib.save(self.image, imagePath)

        # Write out image metadata
        metadataPath = os.path.join(dataDirPath, self.metadataFileName())
        with open(metadataPath, mode='w') as metadataFile:
            # Required fields for BIDS sidecar file
            # Normally, this metadata is defined in the imgMetadata dictionary
            # passed on construction and is derived from the source DICOM
            # TODO(spolcyn): Support volume timing and associated fields
            # TODO(spolcyn): Write out internally used entities, like "subject",
            # "task", and "run" in a suitable way (likely not at all, since
            # they're included in the filename)
            errorMsg = "Metadata didn't contain {}, required by BIDS"
            requiredFields = ["TaskName", "RepetitionTime"]
            for field in requiredFields:
                if not self.imgMetadata.get(field, None):
                    raise RuntimeError(errorMsg.format(field))

            # TODO(spolcyn): Remove after done testing for full bids validation
            self.imgMetadata["SliceTiming"] = list(np.linspace(0.0, 1.5, 27))

            json.dump(self.imgMetadata, metadataFile, sort_keys=True, indent=4)

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
