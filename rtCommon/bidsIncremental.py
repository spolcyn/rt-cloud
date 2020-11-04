"""-----------------------------------------------------------------------------

bidsIncremental.py

This script includes all of the functions that are needed to convert between
DICOM, BIDS-Incremental (BIDS-I), and BIDS formats.

-----------------------------------------------------------------------------"""

from bidsStructures import BidsArchive, BidsIncremental
import logging
import nibabel as nib
import numpy as np
import os
import pydicom
import random
import re

from rtCommon.errors import ValidationError
from rtCommon.imageHandling import convertDicomImgToNifti

logger = logging.getLogger(__name__)


def getMetadata(dicomImg: pydicom.dataset.Dataset) -> (dict, dict):
    """
    Returns the public and private metadata from the provided DICOM image.

    Args:
        dicomImg: A pydicom object to read metadata from.
    Returns:
        Tuple of 2 dictionaries, the first containing the public metadata from
        the image and the second containing the private metadata.
    """
    if not isinstance(dicomImg, pydicom.dataset.Dataset):
        raise ValidationError("Expected pydicom.dataset.Dataset as argument")

    publicMeta = {}
    privateMeta = {}

    # BIDS recommends CamelCase for the key names, which can be obtained from
    # DICOM key names by removing non-alphanumeric characters
    # NOTE: Keys like 'Frame of Reference UID' become 'FrameofReferenceUID',
    # which might be different than the expected behavior
    removalRegex = re.compile('[^a-zA-z]')
    ignoredTags = ['Pixel Data']

    for elem in dicomImg:
        if elem.name in ignoredTags:
            continue

        cleanedKey = removalRegex.sub("", elem.name)
        # in DICOM, public tags have even group numbers and private tags are odd
        # http://dicom.nema.org/dicom/2013/output/chtml/part05/chapter_7.html
        if elem.tag.group % 2 == 0:
            publicMeta[cleanedKey] = str(elem.value)
        else:
            privateMeta[cleanedKey] = str(elem.value)

    return (publicMeta, privateMeta)


def dicomToBidsinc(dicomImg: pydicom.dataset.Dataset) -> BidsIncremental:
    # TODO(spolcyn): Do this all in memory -- dicom2nifti is promising
    # Put extra metadata in sidecar JSON file
    #
    # NOTE: This is currently a method stub pending further dev.
    # The conversion from DICOM to BIDS-I and gathering all required metadata
    # can be complex, as DICOM doesn't necessarily have the metadata required
    # for BIDS in it by default. Thus, another component will handle the logic
    # and error handling surrounding this.
    niftiImage = convertDicomImgToNifti(dicomImg)
    publicMeta, privateMeta = getMetadata(dicomImg)

    publicMeta.update(privateMeta)  # combine metadata dictionaries
    requiredMetadata = {'sub': '002', 'task': 'story', 'contrast_label': 'bold'}
    publicMeta.update(requiredMetadata)
    return BidsIncremental(niftiImage, publicMeta)


def appendBidsinc(incremental: BidsIncremental,
                  archive: BidsArchive,
                  makePath: bool = False) -> None:
    """
    Appends the provided BIDS Incremental imaging and metadata to the provided
    BIDS archive. By default, expects that the incremental represents a valid
    subset of the archive and no additional directory paths will need to be
    created within the archive (this behavior can be overriden).

    Args:
        incremental: BIDS Incremental file containing image data and metadata
        archive: BIDS Archive file to append image data and metadata to
        makePath: Create the directory path for the BIDS-I in the archive if it
            doesn't already exist.

    Returns:
        None

    Raises:
        ValidationError: If the image path within the BIDS incremental does not
            match any existing paths within the archive, and no override is set

    """
    # 1) Create target path for image in archive
    imgDirPath = incremental.makeDataDirPath()
    imgPath = os.path.join(imgDirPath, incremental.getImageFileName())
    imgDirPath = os.path.join(imgDirPath, "")
    logger.debug("Image dir path: %s", imgDirPath)
    logger.debug("Image file path: %s", imgPath)

    # 2) Verify we have a valid way to append the image to the archive. 3 cases:
    # 2.1) Image already exists within archive, append this Nifti to that Nifti
    # 2.2) Image doesn't exist in archive, but rest of the path is valid for the
    # archive; create new Nifti file within the archive
    # 2.3) Neither image nor path is valid for provided archive; fail append
    if archive.pathExists(imgPath):
        logger.debug("Image exists in archive, appending")

        incrementalData = incremental.niftiImg.get_fdata()
        archiveImg = archive.getImage(imgPath)
        archiveData = archiveImg.get_fdata()

        # Build 4-D NIfTI if archive has 3-D, concat to 4-D otherwise
        if len(archiveData.shape) == 3:
            newArchiveData = np.stack((archiveData, incrementalData), axis=3)
        else:
            incrementalData = np.expand_dims(incrementalData, 3)
            newArchiveData = np.concatenate((archiveData, incrementalData), axis=3)

        newImg = nib.Nifti1Image(newArchiveData, archiveImg.affine)
        archive.addImage(newImg, imgPath)

    elif archive.pathExists(imgDirPath):
        logger.debug("Image doesn't exist in archive, creating")

        incrementalData = incremental.niftiImg.get_fdata()
        newImg = nib.Nifti1Image(incrementalData, None)
        archive.addImage(newImg, imgPath)

    else:
        logger.debug("No valid archive path for image, can't append")
        if makePath:
            logger.debug("Overriding no valid archive path, creating dir path")
            incrementalData = incremental.niftiImg.get_fdata()
            newImg = nib.Nifti1Image(incrementalData, None)
            archive.addImage(newImg, imgPath)


def bidsToBidsinc(bidsArchive: BidsArchive,
                  subjectID: str,
                  sessionID: str,
                  dataType: str,
                  otherLabels: str,
                  imageIndex: int):
    """
    Returns a Bids-Incremental file with the specified image of the bidsarchive
    Args:
        bidsArchive: The archive to pull data from
        subjectID: Subject ID to pull data for (for "sub-control01", ID is
            "control01")
        sessionID: Session ID to pull data for (for "ses-2020", ID is "2020")
        dataType: Type of data to pull (common types: anat, func, dwi, fmap).
            This string must be the same as the name of the directory containing
            the image data.
        otherLabels: Other labels specifying appropriate file to pull data for
            (for "sub-control01_task-nback_bold", label is "task-nback_bold").
        imageIndex: Index of image to pull if specifying a 4-D sequence.
    Examples:
        bidsToBidsInc(archive, "01", "2020", "func", "task-nback_bold", 0) will
        extract the first image of the volume at:
        "sub-01/ses-2020/func/sub-01_task-nback_bold.nii"
    """
    pass
