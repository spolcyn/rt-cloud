"""-----------------------------------------------------------------------------

bidsRun.py

Implements the BIDS Run data type used as a buffer for efficiently getting and
appending fMRI runs' data to a BIDS Archive.

-----------------------------------------------------------------------------"""
import logging

from rtCommon.bidsCommon import (
    getNiftiData,
    niftiImagesAppendCompatible,
    symmetricDictDifference,
)
from rtCommon.bidsIncremental import BidsIncremental
from rtCommon.errors import MetadataMismatchError

logger = logging.getLogger(__name__)


class BidsRun:
    def __init__(self, **entities):
        self.incrementals = []
        self._entities = entities

    def __eq__(self, other):
        if self.numIncrementals() == other.numIncrementals():
            if self.getRunEntities() == other.getRunEntities():
                if self.incrementals == other.incrementals:
                    return True
        return False

    def getIncremental(self, index: int) -> BidsIncremental:
        """
        Returns the incremental in the run at the provided index.

        Arguments:
            index: Which image of the run to get (0-indexed)

        Returns:
            Incremental at provided index.

        Raises:
            IndexError: If index is out of bounds for this run.
        """
        try:
            return self.incrementals[index]
        except IndexError:
            raise IndexError(f"Index {index} out of bounds for run with "
                             f"{self.numVols()} incrementals")

    def appendIncremental(self, incremental: BidsIncremental,
                          validateAppend: bool = True) -> None:
        """
        Appends an incremental to this run's data, setting the run's entities if
        the run is empty.

        Arguments:
            incremental: The incremental to add to the run.  validateAppend:
                Validate the incremental matches the current run's data (default
                True). Turning off is useful for efficiently creating a whole
                run at once from an existing image volume, where all data is
                known to be match already.

        Raises:
            MetadataMismatchError: If either the incremental's entities or its
                images's NIfTI header don't match the existing run's data.
        """
        # Set this run's entities if not already present
        if len(self._entities) == 0:
            self._entities = incremental.entities

        if validateAppend:
            if not incremental.entities == self._entities:
                entityDifference = symmetricDictDifference(self._entities,
                                                           incremental.entities)
                errorMsg = ("Incremental's BIDS entities do not match this "
                            f"run's entities (difference: {entityDifference})")
                raise MetadataMismatchError(errorMsg)

            if self.numIncrementals() > 0:
                canAppend, niftiErrorMsg = \
                    niftiImagesAppendCompatible(incremental.image,
                                                self.incrementals[-1].image)

                if not canAppend:
                    errorMsg = ("Incremental's NIfTI header not compatible "
                                f" with this run's images ({niftiErrorMsg})")
                    raise MetadataMismatchError(errorMsg)

        # Slice up the incremental into smaller incrementals if it has multiple
        # images in its image volume
        imagesInVolume = incremental.imageDimensions[3]
        if imagesInVolume == 1:
            self.incrementals.append(incremental)
        else:
            # Split up the incremental into single-image volumes
            image = incremental.image
            imageData = getNiftiData(image)
            affine = image.affine
            header = image.header
            metadata = incremental.imageMetadata

            for imageIdx in range(imagesInVolume):
                newData = imageData[..., imageIdx]
                newImage = incremental.image.__class__(newData, affine, header)
                newIncremental = BidsIncremental(newImage, metadata)
                self.incrementals.append(newIncremental)

    def numIncrementals(self) -> int:
        """
        Returns number of incrementals in this run.
        """
        return len(self.incrementals)

    def getRunEntities(self) -> dict:
        """
        Returns dictionary of the BIDS entities associated with this run.
        """
        return self._entities
