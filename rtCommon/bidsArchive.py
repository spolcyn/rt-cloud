"""-----------------------------------------------------------------------------

bidsArchive.py

Implements interacting with an on-disk BIDS Archive.

-----------------------------------------------------------------------------"""
import json
import logging
import numpy as np
import os
from typing import List

from bids import BIDSLayout
from bids.layout import parse_file_entities as bids_parse_file_entities
from bids.layout.writing import build_path as bids_build_path
import bids.config as bc
import nibabel as nib

from rtCommon.bidsCommon import (
    BIDS_DIR_PATH_PATTERN,
    getNiftiData,
    isJsonPath,
    isNiftiPath,
)
from rtCommon.bidsIncremental import BidsIncremental
from rtCommon import bidsLibrary as bl
from rtCommon.errors import StateError, ValidationError
from rtCommon.imageHandling import readNifti

# Silence future warning
bc.set_option('extension_initial_dot', True)

logger = logging.getLogger(__name__)


class BidsDataset:
    def __init__(self, path: str):
        self.name = os.path.basename(os.path.normpath(path))
        # logger.debug("Loading dataset \"%s\" from: %s", self.name, path)
        self.data = BIDSLayout(path)
        # logger.debug("Dataset info: %s", self.data)

    def __str__(self):
        return str(self.data)

    @staticmethod
    def _stripRoot(relPath: str) -> str:
        """
        Strips a leading / from the path, preventing paths defined relative to
        dataset root (/sub-01/ses-01) from being interpreted as being relative
        to the root of the filesystem
        """
        if len(relPath) >= 1 and relPath[0] == "/":
            return relPath[1:]
        else:
            return relPath

    def absPathFromRelPath(self, relPath: str) -> str:
        """
        Makes an absolute path from the relative path within the dataset.
        """
        return os.path.join(self.data.root, self._stripRoot(relPath))

    def fileExists(self, relPath: str) -> bool:
        # search for matching BIDS file in the dataset
        relPath = self._stripRoot(relPath)
        result = self.data.get_file(self._stripRoot(relPath))
        if result is not None:
            return True

    def dirExists(self, relPath: str) -> bool:
        # search for matching directory in the dataset
        return os.path.isdir(self.absPathFromRelPath(relPath))

    def pathExists(self, path: str) -> bool:
        return self.fileExists(path) or self.dirExists(path)

    def findImage(self, path: str) -> nib.Nifti1Image:
        """
        Find an image within the dataset, if it exists.
        """
        # Validate extension and file existence
        if isNiftiPath(path) and self.fileExists(path):
            return readNifti(self.absPathFromRelPath(path))

        return None

    def _ensurePathExists(self, relPath: str):
        """
        Ensures the provided directory path exists in the dataset, creating
        directories if needed.

        Args:
            relPath: Path to ensure existence of, relative to directory root.
                All parts of the path are assumed to be directories.

        Returns:
            True if a layout update is needed, false otherwise.
        """
        if self.dirExists(relPath):
            return False
        else:
            os.makedirs(self.absPathFromRelPath(relPath))
            return True

    def _updateLayout(self):
        """
        Updates the layout of the dataset so that any new metadata or image
        files are added to the index.
        """
        # TODO(spolcyn): Find if there's a more efficient way to update the
        # index that doesn't rely on implementation details of the PyBids (ie,
        # the SQL DB it uses)
        self.data = BIDSLayout(self.data.root)

    def addImage(self, img: nib.Nifti1Image, path: str) -> None:
        """
        Add an image to the dataset at the provided path, creating the path if
        it does not yet exist.
        """
        logger.debug("Writing new image to %s", self.absPathFromRelPath(path))
        layoutUpdateRequired = self._ensurePathExists(os.path.dirname(path))

        absPath = self.absPathFromRelPath(path)
        nib.save(img, absPath)

        if layoutUpdateRequired:
            self._updateLayout()

    def findMetadata(self, path: str) -> dict:
        """
        Finds metadata for the file at path in the dataset. For an image file,
        this will include the entities embedded in the pathname (e.g, 'subject')
        as well as the metdata found in any sidecar metadata files.

        Args:
            path: Relative path to the file to obtain metdata for.

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
        """
        Finds all files within the dataset at the provided path. Path should be
        relative to dataset root.

        Args:
            path: Path relative to dataset root to search for files in.

        Returns:
            List of absoulte paths on disk to the files found at the target.

        """
        # Get all files
        # TODO(spolcyn): Make more efficient using pybids internal query?
        files = self.data.get_files()
        matchingFiles = []
        absPath = self.absPathFromRelPath(path)

        for _, bidsFile in files.items():
            if bidsFile.path.startswith(absPath):
                matchingFiles.append(bidsFile.path)

        return matchingFiles


def failIfEmpty(func):
    def emptyFailWrapFunction(*args, **kwargs):
        if args[0].dataset is None:
            raise StateError("Dataset empty")
        else:
            return func(*args, **kwargs)

    return emptyFailWrapFunction


class BidsArchive:
    """
    Represents a BIDS Archive
    """
    def __init__(self, rootPath: str):
        self.rootPath = rootPath
        try:
            self.openDataset(self.rootPath)
        except Exception as e:
            logger.info("Failed to open dataset at %s. %s",
                        self.rootPath, str(e))
            self.dataset = None

    def __str__(self):
        return str(self.dataset)

    # Used to update the archive if any on-disk changes have happened
    def _update(self):
        if self.dataset:
            self.dataset._updateLayout()

    @failIfEmpty
    def subjects(self) -> List:
        return self.dataset.data.get_subjects()

    def isEmpty(self) -> bool:
        return (self.dataset is None)

    def openDataset(self, rootPath: str):
        self.dataset = BidsDataset(rootPath)

    @failIfEmpty
    def pathExists(self, path: str) -> bool:
        """
        Check whether the provided path is valid within the archive.
        """
        return self.dataset.pathExists(path)

    @failIfEmpty
    def getFilesForPath(self, path: str) -> List:
        return self.dataset.findFiles(path)

    @failIfEmpty
    def getImage(self, path: str) -> nib.Nifti1Image:
        return self.dataset.findImage(path)

    @failIfEmpty
    def addImage(self, img: nib.Nifti1Image, path: str) -> None:
        # TODO(spolcyn): Change behavior to initialize dafault archive
        # information if enough data is provided in the path to be BIDS
        # compatible
        self.dataset.addImage(img, path)

    @failIfEmpty
    def getMetadata(self, path: str) -> dict:
        return self.dataset.findMetadata(path)

    @failIfEmpty
    def addMetadata(self, metadata: dict, path: str) -> None:
        self.dataset.addMetadata(metadata, path)

    def appendIncremental(self,
                          incremental: BidsIncremental,
                          makePath: bool = True) -> None:
        """
        Appends a BIDS Incremental's image data and metadata to the archive,
        creating new directories if necessary (this behavior can be overridden).

        Args:
            incremental: BIDS Incremental to append
            makePath: Create new directory path for BIDS-I data if needed

        Raises:
            ValidationError: If the image path within the BIDS-I would result in
                directory creation and makePath is set to False.
        """
        # 1) Create target path for image in archive
        dataDirPath = incremental.dataDirPath()
        imgPath = incremental.imageFilePath()
        metadataPath = incremental.metadataFilePath()

        # 2) Verify we have a valid way to append the image to the archive.
        # 3 cases:
        # 2.0) Archive is empty and must be created
        # 2.1) Image already exists within archive, append this Nifti to it
        # 2.2) Image doesn't exist in archive, but rest of the path is valid for
        # the archive; create new Nifti file within the archive
        # 2.3) Neither image nor path is valid for provided archive; fail append
        if self.isEmpty() and makePath:
            incremental.writeToArchive(self.rootPath)
            self.openDataset(self.rootPath)

        elif self.pathExists(imgPath):
            logger.debug("Image exists in archive, appending")
            archiveImg = self.getImage(imgPath)

            # Validate header match
            if not bl.verifyNiftiHeadersMatch(incremental.image,
                                              archiveImg):
                raise ValidationError("Nifti headers failed validation!")
            if not bl.verifyMetadataMatch(incremental.imgMetadata,
                                          self.getMetadata(imgPath)):
                raise ValidationError("Image metadata failed validation!")

            # Build 4-D NIfTI if archive has 3-D, concat to 4-D otherwise
            incrementalData = getNiftiData(incremental.image)
            archiveData = getNiftiData(archiveImg)

            # If archive has something that isn't 3-D or 4-D, not sure what to
            # do with it
            # TODO(spolcyn): Handle this case more gracefully, or provide
            # additional documentation that precludes this from happening
            assert len(archiveData.shape) == 3 or len(archiveData.shape) == 4

            if len(archiveData.shape) == 3:
                archiveData = np.expand_dims(archiveData, 3)

            newArchiveData = np.concatenate((archiveData, incrementalData),
                                            axis=3)

            newImg = nib.Nifti1Image(newArchiveData,
                                     archiveImg.affine,
                                     header=archiveImg.header)
            newImg.update_header()
            self.addImage(newImg, imgPath)

        # 2.2) Image doesn't exist in archive, but rest of the path is valid for
        # the archive; create new Nifti file within the archive
        elif self.pathExists(dataDirPath) or makePath is True:
            logger.debug("Image doesn't exist in archive, creating")

            self.addImage(incremental.image, imgPath)
            self.addMetadata(incremental.imgMetadata, metadataPath)

        else:
            raise ValidationError("No valid archive path for image and no override \
                                   specified, can't append")

    @failIfEmpty
    def stripIncremental(self, subject: str, session: str, task: str,
                         suffix: str, dataType: str, sliceIndex: int = 0,
                         otherLabels: dict = None):
        """
        Creates a BIDS-Incremental file from the specified part of the BIDS
        Archive.

        Args:
            archive: The archive to pull data from
            subject: Subject ID to pull data for (for "sub-control01", ID is
                "control01")
            session: Session ID to pull data for (for "ses-2020", ID is "2020")
            task: Task to pull data for (for "task-nback", name is "nback")
            suffix: BIDS suffix for file, which is image contrast for fMRI
                (bold, cbv, or phase)
            sliceIndex: Index of 3_D image to select in a 4-D sequence of images
            dataType: Type of data to pull (common types: anat, func, dwi, fmap)
                This string must be the same as the name of the directory
                containing the image data.
            otherLabels: Other entity labels specifying appropriate file to pull
                data for (e.g., 'run', 'rec', 'dir', 'echo')

        Returns:
            BIDS-Incremental file with the specified image of the archive and
                its associated metadata

        Examples:
            bidsToBidsInc(archive, "01", "2020", "func", "task-nback_bold", 0)
            will extract the first image of the volume at:
            "sub-01/ses-2020/func/sub-01_task-nback_bold.nii"

        """
        if sliceIndex < 0:
            logger.error(f"Slice index must be >= 0 (got {sliceIndex})")
            return None

        metadata = {'subject': subject, 'session': session, 'task': task,
                    'suffix': suffix, 'datatype': dataType}
        archivePath = bids_build_path(metadata, BIDS_DIR_PATH_PATTERN)

        matchingFilePaths = self.getFilesForPath(archivePath)
        niftiPaths = [path for path in matchingFilePaths if isNiftiPath(path)]
        metaPaths = [path for path in matchingFilePaths if isJsonPath(path)]

        # Fail if no images
        if not niftiPaths:
            logger.error("Archive didn't contain any matching images")
            return None

        # Warn if no metadata
        if not metaPaths:
            logger.warning("Archive didn't contain any matching metadata")

        image = None

        def pathEntitiesMatch(path) -> bool:
            """
            Return true if the BIDS entities contained in the file at the given
            path match the entities provided to the BIDS -> BIDS-I conversion
            method.
            """
            entities = bids_parse_file_entities(path)

            if entities.get("task") != task or \
               entities.get("subject") != subject or \
               entities.get("session") != session:
                return False

            if otherLabels:
                for label, value in otherLabels.items():
                    if entities.get(label) != value:
                        return False

            if suffix and entities.get("suffix") != suffix:
                return False

            return True

        for path in niftiPaths:
            if pathEntitiesMatch(path):
                image = readNifti(path)
                break

        for path in metaPaths:
            if pathEntitiesMatch(path):
                with open(path, 'r', encoding='utf-8') as metadataFile:
                    metadata.update(json.load(metadataFile))
                break

        if image is None:
            logger.error("Failed to find matching image in BIDS Archive for "
                         "provided metadata")
            return None
        elif len(image.dataobj.shape) == 3:
            if sliceIndex != 0:
                logger.error("Matching image was a 3-D NIfTI; time index %d "
                             "too high for a 3-D NIfTI (must be 0)", sliceIndex)
                return None
            return BidsIncremental(image, metadata)
        else:
            slices = nib.four_to_three(image)

            if sliceIndex < len(slices):
                newImage = slices[sliceIndex]
                return BidsIncremental(newImage, metadata)
            else:
                logger.error(f"Image index {sliceIndex} too large for NIfTI "
                             f"volume of length {len(slices)}")
                return None
