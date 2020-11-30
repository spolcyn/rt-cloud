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

from bids.layout.writing import build_path as bidsBuildPath
from bids.layout import parse_file_entities as bidsParseFileEntities
from rtCommon.errors import ValidationError
from rtCommon.bidsCommon import (
    BidsFileExtension,
    BIDS_VERSION,
    DATASET_DESC_REQ_FIELDS,
)

logger = logging.getLogger(__name__)


class BidsIncremental:
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

        if subject is None:
            raise ValidationError("Subject cannot be none")
        if task is None:
            raise ValidationError("Task cannot be none")
        if suffix is None:
            raise ValidationError("Suffix cannot be none")

        self.imgMetadata = imgMetadata

        if datasetMetadata is None:
            datasetMetadata = {}
            datasetMetadata["Name"] = "BIDS-Incremental Dataset"
            datasetMetadata["BIDSVersion"] = str(BIDS_VERSION)
        else:
            missingFields = [field for
                             field in DATASET_DESC_REQ_FIELDS
                             if datasetMetadata.get(field) is None]

            if missingFields != []:
                errorMsg = "Dataset description provided, but missing these \
                        required fields: " + missingFields
                raise ValidationError(errorMsg)

        # The BIDS-I version for serialization
        self.version = 1

    def __str__(self):
        return "Image shape: {}; # Metadata Keys: {}; Version: {}".format(
            self.image.get_fdata().shape,
            len(self.imgMetadata.keys()),
            self.version)

    @staticmethod

    def getFieldLabelString(self, bidsField: str) -> str:
        """
        Extracts the field-label combination for the provided BIDS Standard
        field from this BIDS Incremental. Valid fields are defined in the "Task
        (including resting state) imaging data" section of the BIDS standard,
        and include examples like "sub", "ses", "task", and "run".

        Returns:
            The label (e.g, '01' for 'sub') if present, None otherwise.
        """
        # See if value already cached
        label = self.imgMetadata.get(bidsField, None)
        if label is not None:
            return label

        # Attempt to extract from protocol name metadata if present
        protocolName = self.imgMetadata.get('ProtocolName', None)
        if protocolName is None:
            return None

        prefix = "(?:(?<=_)|(?<=^))"  # match beginning of string or underscore
        suffix = "(?:(?=_)|(?=$))"  # match end of string or underscore

        pattern = "{prefix}(?:{field}-)(.+?){suffix}".format(prefix=prefix,
                                                             field=bidsField,
                                                             suffix=suffix)
        result = re.search(pattern, protocolName)

        if len(result.groups()) == 1:
            return result.group(1)
        else:
            logger.debug("Failed to find exactly one match in protocol name \
                    \'%s\' for field %s", protocolName, bidsField)
            return None

    def getSubjectID(self):
        return self.getFieldLabelString('sub')

    def getSessionName(self):
        return self.getFieldLabelString('ses')

    def getTaskName(self):
        return self.getFieldLabelString('task')

    def getDataTypeName(self):
        # TODO: Support anatomical imaging too
        """ func or anat """
        return "func"

    def getRunLabel(self):
        return self.getFieldLabelString('run')

    def getContrastLabel(self):
        return self.getFieldLabelString('contrast_label')

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
