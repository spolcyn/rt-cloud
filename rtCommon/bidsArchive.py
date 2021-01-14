"""-----------------------------------------------------------------------------

bidsArchive.py

Implements interacting with an on-disk BIDS Archive.

-----------------------------------------------------------------------------"""
import json
import logging
import os
from typing import List

from bids import BIDSLayout
from bids.layout import parse_file_entities as bids_parse_file_entities
from bids.layout.writing import build_path as bids_build_path
import bids.config as bc
import nibabel as nib
import numpy as np

from rtCommon.bidsCommon import (
    BIDS_DIR_PATH_PATTERN,
    getNiftiData,
    isJsonPath,
    isNiftiPath,
)
from rtCommon.bidsIncremental import BidsIncremental
from rtCommon.errors import StateError, ValidationError
from rtCommon.imageHandling import readNifti

# Silence future warning
bc.set_option('extension_initial_dot', True)

logger = logging.getLogger(__name__)


def failIfEmpty(func):
    def emptyFailWrapFunction(*args, **kwargs):
        if args[0].data is None:
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
            self.data = BIDSLayout(rootPath)
        except Exception as e:
            logger.info("Failed to open dataset at %s (%s)",
                        self.rootPath, str(e))
            self.data: BIDSLayout = None

    def __str__(self):
        return str(self.data)

    # Enable accessing underlying BIDSLayout properties without inheritance
    def __getattr__(self, attr):
        return getattr(self.data, attr)

    """ Utility functions """
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
        return os.path.join(self.rootPath, self._stripRoot(relPath))

    def fileExists(self, relPath: str) -> bool:
        if self.data is None:
            return False

        # search for matching BIDS file in the dataset
        relPath = self._stripRoot(relPath)
        result = self.data.get_file(self._stripRoot(relPath))

        return result is not None

    def dirExists(self, relPath: str) -> bool:
        if self.data is None:
            return False
        # search for matching directory in the dataset
        return os.path.isdir(self.absPathFromRelPath(relPath))

    def pathExists(self, path: str) -> bool:
        return self.fileExists(path) or self.dirExists(path)

    @failIfEmpty
    def getImage(self, path: str) -> nib.Nifti1Image:
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
        self.data = BIDSLayout(self.rootPath)

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

    def getFileMetadata(self, path: str) -> dict:
        """
        Gets metadata for the file at path in the dataset. For an image file,
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

    # Used to update the archive if any on-disk changes have happened
    def _update(self):
        if self.data:
            self._updateLayout()

    @failIfEmpty
    def subjects(self) -> List:
        return self.data.get_subjects()

    def isEmpty(self) -> bool:
        return (self.data is None)

    @failIfEmpty
    def getFilesForPath(self, path: str) -> List:
        return self.findFiles(path)

    @failIfEmpty
    def getMetadata(self, path: str) -> dict:
        return self.getFileMetadata(path)

    @staticmethod
    def _imagesAppendCompatible(img1: nib.Nifti1Image, img2: nib.Nifti1Image):
        """
        Verifies that two Nifti image headers match in along a defined set of
        NIfTI header fields which should not change during a continuous fMRI
        scanning session.

        This is primarily intended as a safety check, and does not conclusively
        determine that two images are valid to append to together or are part of
        the same scanning session.

        Args:
            header1: First Nifti header to compare (dict of numpy arrays)
            header2: Second Nifti header to compare (dict of numpy arrays)

        Returns:
            True if the headers match along the required dimensions, false
            otherwise.

        """
        fieldsToMatch = ["intent_p1", "intent_p2", "intent_p3", "intent_code",
                         "dim_info", "datatype", "bitpix", "xyzt_units",
                         "slice_duration", "toffset", "scl_slope", "scl_inter",
                         "qform_code", "quatern_b", "quatern_c", "quatern_d",
                         "qoffset_x", "qoffset_y", "qoffset_z",
                         "sform_code", "srow_x", "srow_y", "srow_z"]

        header1 = img1.header
        header2 = img2.header

        for field in fieldsToMatch:
            v1 = header1.get(field)
            v2 = header2.get(field)

            # Use slightly more complicated check to properly match nan values
            if not (np.allclose(v1, v2, atol=0.0, equal_nan=True)):
                logger.debug("Nifti headers don't match on field: %s \
                             (v1: %s, v2: %s)\n", field, v1, v2)
                return False

        # Two NIfTI headers are append-compatible in 2 cases:
        # 1) Pixel dimensions are exactly equal, and dimensions are equal except
        # for in the final dimension
        # 2) One image has one fewer dimension than the other, and all shared
        # dimensions and pixel dimensions are exactly equal
        dimensionMatch = True

        dimensions1 = header1.get("dim")
        dimensions2 = header2.get("dim")

        nDimensions1 = dimensions1[0]
        nDimensions2 = dimensions2[0]

        pixdim1 = header1.get("pixdim")
        pixdim2 = header2.get("pixdim")

        # Case 1
        if nDimensions1 == nDimensions2:
            pixdimEqual = np.array_equal(pixdim1, pixdim2)
            allButFinalEqual = np.array_equal(dimensions1[:nDimensions1],
                                              dimensions2[:nDimensions2])

            if not (pixdimEqual and allButFinalEqual):
                dimensionMatch = False
        # Case 2
        else:
            dimensionsDifferBy1 = abs(nDimensions1 - nDimensions2) == 1

            nSharedDimensions = min(nDimensions1, nDimensions2)
            # Arrays are 1-indexed as # dimensions is stored in first slot
            sharedDimensionsMatch = \
                np.array_equal(dimensions1[1:nSharedDimensions + 1],
                               dimensions2[1:nSharedDimensions + 1])
            # Arrays are 1-indexed as value used in one method of voxel-to-world
            # coordination translation is stored in the first slot (value should
            # be equal across images)
            sharedPixdimMatch = np.array_equal(pixdim1[:nSharedDimensions + 1],
                                               pixdim2[:nSharedDimensions + 1])

            if not (dimensionsDifferBy1 and sharedDimensionsMatch and
                    sharedPixdimMatch):
                dimensionMatch = False

        if not dimensionMatch:
            logger.debug("Nifti headers not append compatible due to mismatch "
                         "in dimensions and pixdim fields. "
                         "Dim 1: %s | Dim 2 %s\n"
                         "Pixdim 1: %s | Pixdim 2 %s",
                         dimensions1, dimensions2, pixdim1, pixdim2)
            return False

        return True

    @staticmethod
    def _metadataAppendCompatible(meta1: dict, meta2: dict):
        """
        Verifies two metadata dictionaries match in a set of required fields. If
        a field is present in only one or neither of the two dictionaries, this
        is considered a match.

        This is primarily intended as a safety check, and does not conclusively
        determine that two images are valid to append to together or are part of
        the same series.

        Args:
            meta1: First metadata dictionary
            meta2: Second metadata dictionary

        Returns:
            True if all keys that are present in both dictionaries have
            equivalent values, False otherwise.

        """
        matchFields = ["Modality", "MagneticFieldStrength", "ImagingFrequency",
                       "Manufacturer", "ManufacturersModelName",
                       "InstitutionName", "InstitutionAddress",
                       "DeviceSerialNumber", "StationName", "BodyPartExamined",
                       "PatientPosition", "EchoTime",
                       "ProcedureStepDescription", "SoftwareVersions",
                       "MRAcquisitionType", "SeriesDescription", "ProtocolName",
                       "ScanningSequence", "SequenceVariant", "ScanOptions",
                       "SequenceName", "SpacingBetweenSlices", "SliceThickness",
                       "ImageType", "RepetitionTime", "PhaseEncodingDirection",
                       "FlipAngle", "InPlanePhaseEncodingDirectionDICOM",
                       "ImageOrientationPatientDICOM", "PartialFourier"]

        # If either field is None, skip and continue checking other fields
        for field in matchFields:
            value1 = meta1.get(field, None)
            if value1 is None:
                continue

            value2 = meta2.get(field, None)
            if value2 is None:
                continue

            if value1 != value2:
                logger.debug(f"Metadata doesn't match on field: {field}"
                             f"(value 1: {value1}, value 2: {value2}")
                return False

        # These fields should not match between two images for a valid append
        differentFields = ["AcquisitionTime", "AcquisitionNumber"]

        for field in differentFields:
            value1 = meta1.get(field, None)
            if value1 is None:
                continue

            value2 = meta2.get(field, None)
            if value2 is None:
                continue

            if value1 == value2:
                logger.debug(f"Metadata matches (shouldn't) on field: {field}"
                             f"(value 1: {value1}, value 2: {value2}")
                return False

        return True

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
            self._updateLayout()

        elif self.pathExists(imgPath):
            logger.debug("Image exists in archive, appending")
            archiveImg = self.getImage(imgPath)

            # Validate header match
            if not self._imagesAppendCompatible(incremental.image,
                                                archiveImg):
                raise ValidationError("Nifti headers not append compatible")
            if not self._metadataAppendCompatible(incremental.imgMetadata,
                                                  self.getMetadata(imgPath)):
                raise ValidationError("Image metadata not append compatible")

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

            # TODO(spolcyn): Replace this with the patched version of
            # concat_images
            newArchiveData = np.concatenate((archiveData, incrementalData),
                                            axis=3)

            newImg = nib.Nifti1Image(newArchiveData,
                                     archiveImg.affine,
                                     header=archiveImg.header)
            newImg.update_header()
            self.addImage(newImg, imgPath)

        # 2.2) Image doesn't exist in archive, but rest of the path is valid for
        # the archive; create new Nifti file within the archive
        elif self.pathExists(dataDirPath) or makePath:
            logger.debug("Image doesn't exist in archive, creating")

            self.addImage(incremental.image, imgPath)
            self.addMetadata(incremental.imgMetadata, metadataPath)

        else:
            raise StateError("No valid archive path for image and no override"
                             "specified, can't append")

    @failIfEmpty
    def getIncremental(self, subject: str, session: str, task: str, suffix: str,
                       dataType: str, sliceIndex: int = 0,
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
