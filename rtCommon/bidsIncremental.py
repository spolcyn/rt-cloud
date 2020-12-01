"""-----------------------------------------------------------------------------

bidsIncremental.py

Implements the BIDS Incremental data type used for streaming BIDS data between
different applications.

-----------------------------------------------------------------------------"""
from functools import lru_cache
import os
import re
from typing import List

import logging
import nibabel as nib
import numpy as np

from bids.layout.writing import build_path as bidsBuildPath
from bids.layout import parse_file_entities as bidsParseFileEntities
from rtCommon.errors import ValidationError
from rtCommon.bidsCommon import (
    loadBidsEntities,
    BidsEntityKeys,
    BidsFileExtension,
    BIDS_VERSION,
    DATASET_DESC_REQ_FIELDS,
)

logger = logging.getLogger(__name__)



class BidsIncremental:
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
        Initializes a BIDS Incremental object with required data and metadata.

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
            imgMetadata = {}

        imgMetadata["sub"] = subject
        imgMetadata["task"] = task
        imgMetadata["suffix"] = suffix

        self.imgMetadata = imgMetadata

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
        for entityValueDict in cls.ENTITIES.values():
            entity = entityValueDict[BidsEntityKeys.ENTITY_KEY.value]
            entitySearchPattern = patternTemplate.format(field=entity)
            result = re.search(entitySearchPattern, protocolName)

            if result is not None and len(result.groups()) == 1:
                foundEntities[entity] = result.group(1)

        return foundEntities

    # From the BIDS Spec: "A file name consists of a chain of entities, or
    # key-value pairs, a suffix and an extension."
    # Thus, we provide a set of methods to extract these values from the BIDS-I.
    def getEntity(self, entityName) -> str:
        """
        Retrieve the entity value for the provided entity name from this BIDS
        Incremental.

        Args:
            entityName: The name of the BIDS entity to retrieve a value for.
                A list of entity names is provided in the BIDS Standard.

        Returns:
            The entity value, or None if this BIDS incremental doesn't contain a
            value for the entity name.

        """
        if self.ENTITIES.get(entityName) is None:
            errorMsg = "'{}' is not a valid BIDS entity name".format(entityName)
            logger.debug(errorMsg)
            raise ValueError(errorMsg)

    def getSuffix(self) -> str:
        return self.imgMetadata.get(BidsFields.SUFFIX_KEY)

    def getExtension(self) -> str:
        return BidsFileExtension.IMAGE

    # Additional
    def getDataTypeName(self):
        # TODO: Support anatomical imaging too
        """ func or anat """
        return "func"

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

        labelPairs.append('sub-' + self.getSubjectID())

        sesName = self.getSessionName()
        if sesName:
            labelPairs.append('ses-' + sesName)

        labelPairs.append('task-' + self.getTaskName())

        runName = self.getRunLabel()
        if runName:
            labelPairs.append('run-' + runName)

        labelPairs.append(self.getContrastLabel())

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

    def getImageFileName(self) -> str:
        return self.makeBidsFileName(BidsFileExtension.IMAGE)

    def getMetadataFileName(self) -> str:
        return self.makeBidsFileName(BidsFileExtension.METADATA)

    def getDatasetName(self) -> str:
        return self.datasetMetadata["Name"]

    def makeDataDirPath(self) -> str:
        """
        Returns the path to the data directory this incremental's data would be
        in if it were in a full BIDS archive.

        Returns:
            Path string relative to root of the imaginary dataset.
        Examples:
            >>> print(bidsi.makePath())
            /sub-01/ses-2011/anat/
        """
        return os.path.join("",
                            'sub-' + self.getSubjectID(),
                            'ses-' + self.getSessionName(),
                            self.getDataTypeName(),
                            "")

    def makeImageFilePath(self) -> str:
        dataDir = self.makeDataDirPath()
        return os.path.join(dataDir, self.getImageFileName())

    def writeToFile(self, directoryPath: str):
        # Create folder structure -- just func for now
        datasetDir = os.path.join(directoryPath, "dataset")
        try:
            os.mkdir(datasetDir)
        except FileExistsError:
            pass

        subjectName = self.getSubjectID()
        subjectDir = os.path.join(datasetDir, "sub-" + subjectName)
        try:
            os.mkdir(subjectDir)
        except FileExistsError:
            pass

        # get session/task info for directories
        session = self.getSessionName()
        if session:
            session = session.group(0)
            sessionDir = os.path.join(subjectDir, session)
            try:
                os.mkdir(sessionDir)
            except FileExistsError:
                pass

        task = self.getTaskName()
        if task:
            task = task.group(0)

        if session:
            funcDir = os.path.join(sessionDir, "func")
        else:
            funcDir = os.path.join(subjectDir, "func")

        try:
            os.mkdir(funcDir)
        except FileExistsError:
            pass

        # Write out nifti to func folder
        niftiFilename = os.path.join(funcDir, 'sub-{}'.format(subjectName))
        if session:
            niftiFilename += '_' + task
        print(niftiFilename)

        nib.save(self.image, os.path.join(funcDir, niftiFilename))

        # Write out metadata

        # Write out other required files (eg README, dataset description)
