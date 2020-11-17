"""-----------------------------------------------------------------------------

bidsDataset.py

This script implements data structures for representing BIDS-formatted
datasets.

-----------------------------------------------------------------------------"""

from bids import BIDSLayout
import bids.config as bc
from enum import Enum
import json
import logging
import nibabel as nib
import os
import re
from typing import List

from rtCommon.errors import ValidationError
from rtCommon.imageHandling import readNifti
import rtCommon.bidsUtils as bidsUtils

logger = logging.getLogger(__name__)

# Silence future warning
bc.set_option('extension_initial_dot', True)


class BidsDataset:
    def __init__(self, path: str):
        self.name = os.path.basename(os.path.normpath(path))
        logger.debug("Loading dataset \"%s\" from: %s", self.name, path)
        self.data = BIDSLayout(path)
        logger.debug("Dataset info: %s", self.data)

    def __str__(self):
        return str(self.data)

    def insert(self) -> None:
        # Insert some type of data to the dataset
        pass

    def erase(self) -> None:
        # Erase some type of data from the dataset
        pass

    def absPathFromRelPath(self, relPath: str) -> str:
        """
        Makes an absolute path from the relative path within the dataset.
        """
        return os.path.join(self.data.root, relPath)

    def fileExists(self, path: str) -> bool:
        # search for matching BIDS file in the dataset
        result = self.data.get_file(path)
        if result is not None:
            return True

    def dirExists(self, path: str) -> bool:
        # search for matching directory in the dataset
        return os.path.isdir(self.absPathFromRelPath(path))

    def pathExists(self, path: str) -> bool:
        return self.fileExists(path) or self.dirExists(path)

    def findImage(self, path: str) -> nib.Nifti1Image:
        """
        Find an image within the dataset, if it exists.
        """
        # Validate extension and file existence
        if bidsUtils.isNiftiPath(path) and self.fileExists(path):
            return readNifti(self.absPathFromRelPath(path))

        return None

    def _ensurePathExists(self, absPath: str):
        """
        Checks if the provided path exists on disk, creating it if not. Only
        creates directories along the path.

        Returns:
            True if a layout update is needed, false otherwise.
        """
        dirPath = os.path.split(absPath)[0]
        if self.dirExists(dirPath):
            return False

        os.makedirs(dirPath)
        return True

    def _updateLayout(self):
        """
        Updates the layout of the dataset so that any new metadata or image
        files are added to the index.
        """
        # TODO: Find if there's a more efficient way to update the index
        # that doesn't rely on implementation details of the PyBids (ie, the
        # SQLite DB it uses)
        self.data = BIDSLayout(self.data.root)

    def addImage(self, img: nib.Nifti1Image, path: str) -> None:
        """
        Add an image to the dataset at the provided path, creating the path if
        it does not yet exist.
        """
        logger.debug("Writing new image to %s", self.absPathFromRelPath(path))
        absPath = self.absPathFromRelPath(path)
        layoutUpdateRequired = self._ensurePathExists(absPath)

        nib.save(img, absPath)

        if layoutUpdateRequired:
            self._updateLayout()

    def findMetadata(self, path: str) -> dict:
        """
        Finds metadata for the file at path in the dataset.

        Args:
            path: Relative path to an image file.

        Returns:
            Dictionary with sidecar metadata for the file and any metadata that
                can be extracted from the filename (e.g., subject, session)

        """
        return self.data.get_metadata(self.absPathFromRelPath(path),
                                      include_entities=True)

    def addMetadata(self, metadata: dict, path: str) -> None:
        absPath = self.absPathFromRelPath(path)

        with open(absPath, 'w', encoding='utf-8') as metadataFile:
            json.dump(metadata, metadataFile, ensure_ascii=False, indent=4)

        logger.debug("Wrote new metadata to %s", absPath)

        self._updateLayout()

    def findFiles(self, path: str) -> List:
        # Get all files
        # TODO: Make more efficient using pybids internal query?
        files = self.data.get_files()
        # logger.debug("Files: %s", files)
        matchingFiles = []
        absPath = self.absPathFromRelPath(path)
        logger.debug("Abs Path: %s\n", absPath)

        for _, v in files.items():
            """
            logger.debug("File: %s\n", v.filename)
            logger.debug("Dir: %s\n", v.dirname)
            logger.debug("Path: %s\n", v.path)
            """
            if v.path.startswith(absPath):
                matchingFiles.append(v.path)

        for f in matchingFiles:
            logger.debug("Match: %s\n", f)

        return matchingFiles


class BidsArchive:
    """
    Represents a BIDS Archive
    """
    def __init__(self, rootPath: str, datasetName: str = None):
        self.dataset = BidsDataset(rootPath)

    def __str__(self):
        return str(self.dataset)

    def pathExists(self, path: str) -> bool:
        """
        Check whether the provided path is valid within the archive.
        """
        return self.dataset.pathExists(path)

    def getImage(self, path: str) -> nib.Nifti1Image:
        return self.dataset.findImage(path)

    def addImage(self, img: nib.Nifti1Image, path: str) -> None:
        self.dataset.addImage(img, path)

    def getMetadata(self, path: str) -> dict:
        return self.dataset.findMetadata(path)

    def addMetadata(self, metadata: dict, path: str) -> None:
        self.dataset.addMetadata(metadata, path)


class BidsFileExtension(Enum):
    IMAGE = '.nii'
    IMAGE_COMPRESSED = '.nii.gz'
    METADATA = '.json'


class BidsIncremental:
    """
    BIDS Incremental data format suitable for streaming BIDS Archives
    """
    def __init__(self,
                 niftiImg: nib.Nifti1Image,
                 imgMetadata: dict,
                 datasetMetadata: dict = None):
        """
        Initializes a BIDS Incremental object with required data and metadata.

        Args:
            niftiImg: Single Nifti image for this BIDS-I as an NiBabel
                NiftiImage.
            imgMetadata: Required BIDS metadata for the NIfTI filename. Required
                fields: 'sub', 'task', 'contrast_type'.
            datasetMetadata: Top-level dataset metadata for the BIDS dataset
                this represents.

        Raises:
            ValidationError: If any required metadata field is not provided.

        """

        # ensure image file only has one image
        imgShape = niftiImg.get_fdata().shape
        assert len(imgShape) == 3 or (len(imgShape) == 4 and imgShape[0] == 1)

        self.image = niftiImg
        self.header = niftiImg.header

        missingMetadata = self.findMissingMetadata(imgMetadata)
        if missingMetadata != []:
            errorMsg = "Image metadata missing these required fields: {}" \
                       .format(missingMetadata)
            raise ValidationError(errorMsg)

        self.imgMetadata = imgMetadata
        self.datasetMetadata = datasetMetadata

        self.version = 1

    @staticmethod
    def findMissingMetadata(metadata: dict) -> List[str]:
        requiredMetadata = ['sub', 'task', 'contrast_label']
        return [key for key in requiredMetadata if key not in metadata]

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
        square brackets [] are considered optional):

        sub-<label>[_ses-<label>]_task-<label>[_acq-<label>]\
        [_ce-<label>] [_dir-<label>][_rec-<label>][_run-<index>]\
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
        return 'dataset'

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
