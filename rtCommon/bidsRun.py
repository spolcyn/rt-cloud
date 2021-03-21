"""-----------------------------------------------------------------------------

bidsRun.py

Implements the BIDS Run data type used as a buffer for efficiently getting and
appending fMRI runs' data to a BIDS Archive.

-----------------------------------------------------------------------------"""
import logging

from rtCommon.bidsCommon import (
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
                          unsafeAppend: bool = False) -> None:
        """
        Appends an incremental to this run's data, setting the run's entities if
        the run is empty.

        Arguments:
            incremental: The incremental to add to the run.
            unsafeAppend: Skip validating whether the incremental matches the
                current run's data (default False). Useful for efficiently
                creating a whole run at once from an existing image volume,
                where all data is known to be match already.

        Raises:
            MetadataMismatchError: If either the incremental's entities or its
                images's NIfTI header don't match the existing run's data.
        """
        # Set this run's entities if not already present
        if len(self._entities) == 0:
            self._entities = incremental.entities

        # No validation needed if first incremental
        if len(self.incrementals) == 0 or unsafeAppend:
            self.incrementals.append(incremental)
            return

        # If run already started, ensure incremental matches run before
        # appending it
        if not incremental.entities == self._entities:
            entityDifference = symmetricDictDifference(self._entities,
                                                       incremental.entities)
            errorMsg = ("Incremental's BIDS entities do not match this run's "
                        f"entities (difference: {entityDifference})")
            raise MetadataMismatchError(errorMsg)

        canAppend, niftiErrorMsg = \
            niftiImagesAppendCompatible(incremental.image,
                                        self.incrementals[-1].image)

        if not canAppend:
            errorMsg = ("Incremental's NIfTI header isn't append-compatible "
                        f"with this run's images ({niftiErrorMsg})")
            raise MetadataMismatchError(errorMsg)

        # All checks passed
        self.incrementals.append(incremental)

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
