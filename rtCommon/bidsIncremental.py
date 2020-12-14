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
    # TODO: Rename this to something more useful
    ENTITIES = loadBidsEntities()

    """
    BIDS Incremental data format suitable for streaming BIDS Archives
    """
    def __init__(self,
                 image: nib.Nifti1Image,
                 subject: str,
                 task: str,
                 suffix: str,
                 imgMetadata: dict = None,
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
        if image is None:
            raise ValidationError("Image cannot be None")
        elif not (isinstance(image, nib.Nifti1Image) or
                  isinstance(image, nib.Nifti2Image)):
            raise ValidationError("Image must be NIBabel Nifti 1 or 2 image, \
                                   got type %s" % str(type(image)))
        else:
            self.image = image
            self.header = image.header

        # Validate and store image metadata
        if subject is None:
            raise ValidationError("Subject cannot be none")
        if task is None:
            raise ValidationError("Task cannot be none")
        if suffix is None:
            raise ValidationError("Suffix cannot be none")

        if imgMetadata is None:
            self.imgMetadata = {}
        else:
            self.imgMetadata = imgMetadata

        self.imgMetadata["subject"] = subject
        self.imgMetadata["task"] = task
        self.imgMetadata["suffix"] = suffix

        protocolName = self.imgMetadata.get("ProtocolName")
        if protocolName is not None:
            self.imgMetadata.update(
                self.parseBidsFieldsFromProtocolName(protocolName))

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
            datasetMetadata["Name"] = "BIDS-Incremental Dataset from RT-Cloud"
            datasetMetadata["BIDSVersion"] = str(BIDS_VERSION)

        self.datasetMetadata = datasetMetadata

        # The BIDS-I version for serialization
        self.version = 1

    def __str__(self):
        return "Image shape: {}; # Metadata Keys: {}; Version: {}".format(
            self.image.get_fdata().shape,
            len(self.imgMetadata.keys()),
            self.version)

    def __eq__(self, other):
        # Compare images
        # TODO: Could do this more completely by save()'ing the images to disk,
        # then diffing the files that they produce
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
    def parseBidsFieldsFromProtocolName(cls, protocolName: str) -> dict:
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
        fieldPat = "(?:{field}-)(.+?)"  # TODO: Document this regex
        patternTemplate = prefix + fieldPat + suffix

        foundEntities = {}
        for entityName, entityValueDict in cls.ENTITIES.items():
            entity = entityValueDict[bek.ENTITY_KEY.value]
            entitySearchPattern = patternTemplate.format(field=entity)
            result = re.search(entitySearchPattern, protocolName)

            if result is not None and len(result.groups()) == 1:
                foundEntities[entityName] = result.group(1)

        return foundEntities

    # TODO: Add specific getters for commonly used things, like getRun,
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
        # TODO: Support anatomical imaging too
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

    # TODO: Write a BIDS-I to a valid BIDS Archive on disk
    # TODO: Call BIDS Validator on that BIDS Archive
    def writeToArchive(self, directoryPath: str):
        """
        Args:
            directoryPath: Location to write the BIDS-I derived BIDS Archive
        """
        # Build path to the data directory
        # Overall hierarchy:
        # sub-<participant_label>/[/ses-<session_label>]/<data_type>/
        pathElements = [self.datasetName(), 'sub-' + self.getEntity("subject")]

        session = self.getEntity("session")
        if session:
            pathElements.append('ses-' + session)

        pathElements.append(self.dataType())

        dataDirPath = os.path.join(directoryPath, *pathElements)
        os.makedirs(dataDirPath, exist_ok=True)

        # Write image to data folder
        imagePath = os.path.join(dataDirPath, self.imageFileName())
        nib.save(self.image, imagePath)

        # Write out metadata
        metadataPath = os.path.join(dataDirPath, self.metadataFileName())
        with open(metadataPath, mode='w') as metadataFile:
            json.dump(self.imgMetadata, metadataFile)

        # TODO: Write out other required files (eg README, dataset description)

    """ END BIDS-I ARCHIVE EMULTATION API """
