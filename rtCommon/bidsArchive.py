"""-----------------------------------------------------------------------------

bidsArchive.py

Implements interacting with an on-disk BIDS Archive.

-----------------------------------------------------------------------------"""
import json
import logging
import os
from typing import List

from bids import BIDSLayout
from bids.exceptions import BIDSValidationError
import bids.config as bc
import nibabel as nib

import rtCommon.bidsCommon as bidsUtils
from rtCommon.bidsIncremental import BidsIncremental
from rtCommon.imageHandling import readNifti

logger = logging.getLogger(__name__)

# Silence future warning
bc.set_option('extension_initial_dot', True)


class BidsDataset:
    def __init__(self, path: str):
        self.name = os.path.basename(os.path.normpath(path))
        # logger.debug("Loading dataset \"%s\" from: %s", self.name, path)
        self.data = BIDSLayout(path)
        # logger.debug("Dataset info: %s", self.data)

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
        # TODO(spolcyn): Find if there's a more efficient way to update the index
        # that doesn't rely on implementation details of the PyBids (ie, the
        # SQL DB it uses)
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

    def isEmpty(self) -> bool:
        return (self.dataset is None)

    def openDataset(self, rootPath: str):
        self.dataset = BidsDataset(rootPath)

    def pathExists(self, path: str) -> bool:
        """
        Check whether the provided path is valid within the archive.
        """
        return self.dataset.pathExists(path)

    def getFilesForPath(self, path: str) -> List:
        return self.dataset.findFiles(path)

    def getImage(self, path: str) -> nib.Nifti1Image:
        return self.dataset.findImage(path)

    def addImage(self, img: nib.Nifti1Image, path: str) -> None:
        self.dataset.addImage(img, path)

    def getMetadata(self, path: str) -> dict:
        return self.dataset.findMetadata(path)

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

        # 2) Verify we have a valid way to append the image to the archive. 3 cases:
        # 2.0) Archive is empty and must be created
        # 2.1) Image already exists within archive, append this Nifti to that Nifti
        # 2.2) Image doesn't exist in archive, but rest of the path is valid for the
        # archive; create new Nifti file within the archive
        # 2.3) Neither image nor path is valid for provided archive; fail append
        if self.isEmpty():
            incremental.writeToArchive(self.rootPath)
            self.openDataset(self.rootPath)

        elif self.pathExists(imgPath):
            logger.debug("Image exists in archive, appending")
            archiveImg = self.getImage(imgPath)

            # Validate header match
            if not verifyNiftiHeadersMatch(incremental.image,
                                           archiveImg):
                raise ValidationError("Nifti headers failed validation!")
            if not verifyMetadataMatch(incremental.imgMetadata,
                                       self.getMetadata(metadataPath)):
                raise ValidationError("Image metadata failed validation!")

            # Build 4-D NIfTI if archive has 3-D, concat to 4-D otherwise
            incrementalData = incremental.image.get_fdata()
            archiveData = archiveImg.get_fdata()

            if len(archiveData.shape) == 3:
                newArchiveData = np.stack((archiveData, incrementalData), axis=3)
            else:
                incrementalData = np.expand_dims(incrementalData, 3)
                newArchiveData = np.concatenate((archiveData, incrementalData),
                                                axis=3)

            newImg = nib.Nifti1Image(newArchiveData,
                                     archiveImg.affine,
                                     header=archiveImg.header)
            newImg.update_header()
            self.addImage(newImg, imgPath)

        elif self.pathExists(imgDirPath) or makePath is True:
            logger.debug("Image doesn't exist in archive, creating")
            self.addImage(incremental.image, imgPath)
            self.addMetadata(incremental.imgMetadata, metadataPath)

        else:
            raise ValidationError("No valid archive path for image and no override \
                                   specified, can't append")
