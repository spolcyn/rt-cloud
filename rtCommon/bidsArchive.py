"""-----------------------------------------------------------------------------

bidsArchive.py

Implements interacting with an on-disk BIDS Archive.

-----------------------------------------------------------------------------"""
import functools
import json
import logging
import os
import re
from typing import List

from bids.config import set_option as bc_set_option
from bids.exceptions import (
    NoMatchError,
)
from bids.layout import (
    BIDSDataFile,
    BIDSFile,
    BIDSImageFile,
    BIDSLayout,
)
import nibabel as nib
import numpy as np

from rtCommon.bidsCommon import (
    getNiftiData,
)
from rtCommon.bidsIncremental import BidsIncremental
from rtCommon.errors import MissingMetadataError, StateError

# Silence future warning
bc_set_option('extension_initial_dot', True)

logger = logging.getLogger(__name__)


def failIfEmpty(func):
    @functools.wraps(func)
    def emptyFailWrapFunction(*args, **kwargs):
        if args[0].data is None:
            raise StateError("Dataset empty")
        else:
            return func(*args, **kwargs)

    return emptyFailWrapFunction


class BidsArchive:

    def __init__(self, rootPath: str):
        """
        BidsArchive represents a BIDS-formatted dataset on disk. It offers an
        API for querying that dataset, and also adds special methods to add
        BidsIncrementals to the dataset and extract portions of the dataset as
        BidsIncrementals.

        Args:
            rootPath: Path to the archive on disk (either absolute or relative
            to current working directory).

        Examples:
            >>> archive = BidsArchive('dataset')
            >>> str(archive)
            Root: ...t-cloud/docs/tutorials/dataset | Subjects: 1 |
            Sessions: 0 | Runs: 1
            >>> archive = BidsArchive('/tmp/downloads/dataset')
            >>> str(archive)
            Root: /tmp/downloads/dataset | Subjects: 20 |
            Sessions: 3 | Runs: 2
        """
        self.rootPath = os.path.abspath(rootPath)
        # Formatting initialization logic this way enables the creation of an
        # empty BIDS archive that an incremntal can then be appended to
        try:
            self.data = BIDSLayout(rootPath)
        except Exception as e:
            logger.debug("Failed to open dataset at %s (%s)",
                         self.rootPath, str(e))
            self.data: BIDSLayout = None

    def __str__(self):
        out = str(self.data)
        if 'BIDS Layout' in out:
            out = out.replace('BIDS Layout', 'Root')

        return out

    # Enable accessing underlying BIDSLayout properties without inheritance
    def __getattr__(self, attr):
        originalAttr = attr

        # If the attr is in the format getXyz, convert to get_xyz for forwarding
        # to the BIDSLayout object
        pattern = re.compile("get[A-Z][a-z]+")
        if pattern.match(attr) is not None:
            attr = attr.lower()
            attr = attr[0:3] + '_' + attr[3:]

        if not self.isEmpty():
            try:
                return getattr(self.data, attr)
            except AttributeError:
                raise AttributeError("{} object has no attribute {}".format(
                    self.__class__.__name__, originalAttr))

    """ Utility functions """
    @staticmethod
    def _stripLeadingSlash(path: str) -> str:
        """
        Strips a leading / from the path, if it exists. This prevents paths
        defined relative to dataset root (/sub-01/ses-01) from being interpreted
        as being relative to the root of the filesystem.

        Args:
            path: Path to strip leading slash from.

        Examples:
            >>> path = '/sub-01/ses-01/func/sub-01_task-test_bold.nii.gz'
            >>> BidsArchive._stripLeadingSlash(path)
            'sub-01/ses-01/func/sub-01_task-test_bold.nii.gz'
            >>> path = 'sub-01/ses-01/func/sub-01_task-test_bold.nii.gz'
            'sub-01/ses-01/func/sub-01_task-test_bold.nii.gz'
        """
        if len(path) >= 1 and path[0] == "/":
            return path[1:]
        else:
            return path

    def absPathFromRelPath(self, relPath: str) -> str:
        """
        Makes an absolute path from the relative path within the dataset.
        """
        return os.path.join(self.rootPath, self._stripLeadingSlash(relPath))

    def fileExistsInArchive(self, path: str) -> bool:
        if self.data is None:
            return False
        else:
            path = self._stripLeadingSlash(path)
            fileExists = self.data.get_file(path) is not None
            logger.debug("File %s exists: %s", path, fileExists)
            return fileExists

    def dirExists(self, relPath: str) -> bool:
        if self.data is None:
            return False
        # search for matching directory in the dataset
        return os.path.isdir(self.absPathFromRelPath(relPath))

    def pathExists(self, path: str) -> bool:
        return self.fileExistsInArchive(path) or self.dirExists(path)

    @failIfEmpty
    def getImages(self, matchExact: bool = False,
                  **entities) -> List[BIDSImageFile]:
        """
        Return all images that have the provided entities. If no entities are
        provided, then all images are returned.

        Args:
            matchExact: Only return images that have exactly the provided
                entities, no more and no less.
            **entities: Entities that returned images must have.

        Returns:
            A list of images matching the provided entities (empty if there are
            no matches, and containing at most a single image if an exact match
            is requested).

        Examples:
            >>> archive = BidsArchive('/path/to/archive')

            Using a dictionary to provide target entities.

            >>> entityDict = {'subject': '01', 'datatype': 'func'}
            >>> images = archive.getImages(**entityDict)

            Using keyword arguments to provide target entities.

            >>> images = archive.getImages(subject='01', datatype='func')

            Accessing properties of the image.

            >>> image = images[0]
            >>> print(image.get_image()
            (64, 64, 27, 3)
            >>> print(image.path)
            /tmp/archive/func/sub-01_task-test_bold.nii
            >>> print(image.filename)
            sub-01_task-test_bold.nii

            An exact match must have exactly the same entities; since images
            must also have the task entity in their filename, the above
            entityDict will yield no exact matches in the archive.

            >>> images = archive.getImages(entityDict, matchExact=True)
            ERROR "No images were an exact match for: {'subject': '01',
            'datatype': 'func'}"
            >>> print(len(images))
            0
        """
        # Validate image extension specified
        extension = entities.pop('extension', None)
        if extension is not None:
            if extension != '.nii' and extension != '.nii.gz':
                raise ValueError('Extension for images must be either .nii or '
                                 '.nii.gz')

        results = self.data.get(**entities)
        results = [r for r in results if type(r) is BIDSImageFile]

        if len(results) == 0:
            logger.error(f"No images have all provided entities: {entities}")
            return []
        elif matchExact:
            for result in results:
                # Only BIDSImageFiles are checked, so extension is irrelevant
                result_entities = result.get_entities()
                result_entities.pop('extension', None)

                if result_entities == entities:
                    return [result]

            logger.error(f"No images were an exact match for: {entities}")
            return []
        else:
            return results

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
        self.data = BIDSLayout(self.rootPath)

    def _addImage(self, img: nib.Nifti1Image, path: str) -> None:
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

    def _addMetadata(self, metadata: dict, path: str) -> None:
        absPath = self.absPathFromRelPath(path)

        with open(absPath, 'w', encoding='utf-8') as metadataFile:
            json.dump(metadata, metadataFile, ensure_ascii=False, indent=4)

        logger.debug("Wrote new metadata to %s", absPath)

        self._updateLayout()

    # Used to update the archive if any on-disk changes have happened
    def _update(self):
        if self.data:
            self._updateLayout()

    def isEmpty(self) -> bool:
        return (self.data is None)

    def tryGetFile(self, path: str) -> BIDSFile:
        """
        Tries to get a file from the archive using different interpretations of
        the target path. Interpretations considered are:
        1) Path with leading slash, relative to filesystem root
        2) Path with leading slash, relative to archive root
        3) Path with no leading slash

        Args:
            path: Path to the file to attempt to get.

        Returns:
            BIDSFile (or subclass) if a matching file was found, None otherwise.

        Examples:
            >>> archive = BidsArchive('/path/to/archive')
            >>> filename = 'sub-01_task-test_bold.nii.gz'
            >>> archive.tryGetFile('/tmp/archive/sub-01/func/' + filename)
            <BIDSImageFile filename=/tmp/archive/sub-01/func/sub-01_task-test\
            _bold.nii.gz
            >>> archive.tryGetFile('/' + filename)
            <BIDSImageFile filename=/tmp/archive/sub-01/func/sub-01_task-test\
            _bold.nii.gz
            >>> archive.tryGetFile(filename)
            <BIDSImageFile filename=/tmp/archive/sub-01/func/sub-01_task-test\
            _bold.nii.gz
        """
        # Tries absolute to disk root and relative to archive root
        archiveFile = self.get_file(path)
        if archiveFile is not None:
            return archiveFile

        # Try absolute to archive root
        strippedRootPath = self._stripLeadingSlash(path)
        archiveFile = self.get_file(strippedRootPath)
        if archiveFile is not None:
            return archiveFile

        return None

    @failIfEmpty
    def getMetadata(self, path: str, includeEntities: bool = True) -> dict:
        """
        Get metadata for the file at the provided path in the dataset. Sidecar
        metadata is always returned, and BIDS entities present in the filename
        are returned by default (this can be disabled).

        Args:
            path: Path to image file to get metadata for
            includeEntities: True to include the entities in the filename (e.g.,
                'subject', 'task', and 'session'), False to include only the
                metadata in sidecar files. Defaults to True.

        Returns:
            Dictionary with sidecar metadata for the file and any metadata that
                can be extracted from the filename (e.g., subject, session).

        Examples:
            >>> archive = BidsArchive('/path/to/archive')
            >>> path = archive.getImages()[0].path
            >>> archive.getMetadata(path)
            {'AcquisitionMatrixPE': 320, 'AcquisitionNumber': 1, ... }
        """
        target = self.tryGetFile(path)
        if target is None:
            raise NoMatchError("File doesn't exist, can't get metadata")
        else:
            return self.data.get_metadata(target.path,
                                          include_entities=includeEntities)

    @failIfEmpty
    def getEvents(self, matchExact: bool = False,
                  **entities) -> List[BIDSDataFile]:
        """
        Gets data from scanner run event files in the archive. Event files to
        retrieve can be filtered by entities present in the files' names.

        Args:
            matchExact: Whether to only return events files that have exactly
                the same entities as provided (no more, no less)
            entities: Keyword arguments for entities to filter by. Provide in
                the format entity='value'.

        Returns:
            A list of BIDSDataFile objects encapsulating the events files
            matching the provided entities (empty if there are no matches, and
            containing at most a single object if an exact match is requested).

        Raises:
            ValueError: If the 'extension' entity is provided and not valid for
                an events file (i.e., not '.tsv' or '.tsv.gz')

        Examples:
            >>> archive = BidsArchive('.')
            >>> archive.getEvents()
            [<BIDSDataFile filename='/tmp/dataset/sub-01/func/\
            sub-01_task-test_events.tsv'>, <BIDSDataFile
            filename='/tmp/dataset/sub-02/func/sub-02_task-test_events.tsv'>]
            >>> sub1Events = archive.getEvents(subject='01')
            [<BIDSDataFile filename='/tmp/dataset/sub-01/func/\
            sub-01_task-test_events.tsv'>]
            >>> eventsDataFrame = sub1Events[0].get_df()
            >>> print(eventsDataFrame[:][:1])
                onset   duration    trial_type
            0   0       30          rest
        """
        # Validate image extension specified
        validExtensions = ['.tsv', '.tsv.gz']
        extension = entities.get('extension', None)
        if extension is not None and extension not in validExtensions:
            raise ValueError(f'Extension must be one of {validExtensions}')

        entities['suffix'] = 'events'

        results = self.data.get(**entities)

        if len(results) == 0:
            logger.error(f"No event files have all provided entities: "
                         f"{entities}")
            return []
        elif matchExact:
            for result in results:
                if result.get_entities() == entities:
                    return [result]

            logger.error(f"No event files were an exact match for: {entities}")
            return []
        else:
            return results

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
                          makePath: bool = True) -> bool:
        """
        Appends a BIDS Incremental's image data and metadata to the archive,
        creating new directories if necessary (this behavior can be overridden).

        Args:
            incremental: BIDS Incremental to append
            makePath: Create new directory path for BIDS-I data if needed.
                (default: True).

        Raises:
            RuntimeError: If the image to append to in the archive is not either
                3D or 4D.
            StateError: If the image path within the BIDS-I would result in
                directory creation and makePath is set to False.
            ValidationError: If the data to append is incompatible with existing
                data in the archive.

        Returns:
            True if the append succeeded, False otherwise.

        Examples:
            Assume we have a NIfTI image 'image' and a metadata dictionary
            'metdata' with all required metadata for a BIDS Incremental.

            >>> archive = BidsArchive('.')
            >>> incremental = BidsIncremental(image, metadata)
            >>> archive.appendIncremental(incremental)

            If we don't want to create any new files/directories in the archive,
            makePath can be set to false.

            >>> archive = BidsArchive('/tmp/emptyDirectory')
            >>> archive.appendIncremental(incremental, makePath=False)
            False
        """
        # 1) Create target path for image in archive
        dataDirPath = incremental.dataDirPath
        imgPath = incremental.imageFilePath
        metadataPath = incremental.metadataFilePath

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

            # Supplement entity dict with file name entities that the BIDSLayout
            # returns that aren't official bids entities
            entityDict = incremental.entities
            entityDict['datatype'] = incremental.datatype
            entityDict['suffix'] = incremental.suffix

            # The one exact match must exist because the path exists
            images = self.getImages(**entityDict, matchExact=True)
            assert len(images) == 1
            archiveImg = images[0].get_image()

            # Validate header match
            if not self._imagesAppendCompatible(incremental.image,
                                                archiveImg):
                raise RuntimeError("Nifti headers not append compatible")
            if not self._metadataAppendCompatible(incremental.imageMetadata,
                                                  self.getMetadata(imgPath)):
                raise RuntimeError("Image metadata not append compatible")

            # Build 4-D NIfTI if archive has 3-D, concat to 4-D otherwise
            incrementalData = getNiftiData(incremental.image)
            archiveData = getNiftiData(archiveImg)

            # RT-Cloud assumes 3D or 4D NIfTI images, other sizes have unknown
            # interpretations
            dimensions = len(archiveData.shape)
            if dimensions == 3:
                archiveData = np.expand_dims(archiveData, 3)
                archiveImg.header['pixdim'][4] = \
                    incremental.getMetadataField('RepetitionTime')
            elif dimensions == 4:
                pass
            else:
                raise RuntimeError("Expected image to have 3 or 4 dimensions "
                                   f"(got {dimensions} dimensions)")

            # TODO(spolcyn): Replace this with Nibabel's concat_images function
            # when the dtype issue with save/load cycle is fixed
            # https://github.com/nipy/nibabel/issues/986
            newArchiveData = np.concatenate((archiveData, incrementalData),
                                            axis=3)

            newImg = nib.Nifti1Image(newArchiveData,
                                     archiveImg.affine,
                                     header=archiveImg.header)
            newImg.update_header()
            self._addImage(newImg, imgPath)

        # 2.2) Image doesn't exist in archive, but rest of the path is valid for
        # the archive; create new Nifti file within the archive
        elif self.pathExists(dataDirPath) or makePath:
            logger.debug("Image doesn't exist in archive, creating")
            self._addImage(incremental.image, imgPath)
            self._addMetadata(incremental.imageMetadata, metadataPath)

        # Archive wasn't empty, specified paths didn't exist, and no permission
        # given to make the path; append fails.
        else:
            return False

        return True

    @failIfEmpty
    def getIncremental(self, sliceIndex: int = 0, **entities) \
            -> BidsIncremental:
        """
        Creates a BIDS Incremental from the specified part of the archive.

        Args:
            sliceIndex: Index of 3-D image to select in a 4-D image volume.
            entities: Keyword arguments for entities to filter by. Provide in
                the format entity='value'.

        Returns:
            BIDS-Incremental file with the specified image of the archive and
            its associated metadata.

        Raises:
            IndexError: If the provided sliceIndex goes beyond the bounds of the
                image specified in the archive.
            MissingMetadataError: If the archive lacks the required metadata to
                make a BIDS Incremental out of an image in the archive.
            NoMatchError: When no images that match the provided entities are
                found in the archive
            RuntimeError:
                1) When too many images that match the provided entities
                are found in the archive.
                2) If the image matching the provided entities has fewer
                than 3 dimensions or greater than 4.

        Examples:
            >>> archive = BidsArchive('.')
            >>> inc = archive.getIncremental(subject='01', task='test')
            >>> entityFilterDict = {'subject': '01', 'task': 'test'}
            >>> inc2 = archive.getIncremental(**entityFilterDict)
            >>> inc == inc2
            True
            >>> inc.imageDimensions
            (64, 64, 27, 1)
        """
        if sliceIndex < 0:
            raise IndexError(f"Slice index must be >= 0 (got {sliceIndex})")

        candidates = self.getImages(**entities)

        # Throw error if not exactly one match
        if len(candidates) == 0:
            raise NoMatchError("Unable to find any data in archive that matches"
                               f" all provided entities: {entities}")
        elif len(candidates) > 1:
            raise RuntimeError("Too many results for entities (expected 1, got "
                               f"{len(candidates)})")

        # Create BIDS-I
        candidate = candidates[0]
        image = candidate.get_image()

        # Process error conditions and slice image if necessary
        if len(image.dataobj.shape) == 3:
            if sliceIndex != 0:
                raise IndexError(f"Matching image was a 3-D NIfTI; {sliceIndex}"
                                 f" too high for a 3-D NIfTI (must be 0)")
        elif len(image.dataobj.shape) == 4:
            slices = nib.four_to_three(image)

            if sliceIndex < len(slices):
                image = slices[sliceIndex]
            else:
                raise IndexError(f"Image index {sliceIndex} too large for NIfTI"
                                 f" volume of length {len(slices)}")
        else:
            raise RuntimeError("Expected image to have 3 or 4 dimensions (got "
                               + len(image.dataobj.shape) + " dimensions)")

        metadata = self.data.get_metadata(candidate.path, include_entities=True)
        metadata.pop('extension')  # unused by BIDS-I

        try:
            return BidsIncremental(image, metadata)
        except MissingMetadataError as e:
            raise MissingMetadataError("Archive lacks required metadata for "
                                       "BIDS Incremental creation: " + str(e))
