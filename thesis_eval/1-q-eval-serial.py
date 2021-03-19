"""
Tests the timing for serialization & deserialization as function of nifti image size
"""

import os
import pickle
import sys
import time

import numpy as np

rtCloudDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(rtCloudDir)
sys.path.append(rtCloudDir)
from rtCommon.bidsIncremental import BidsIncremental
from rtCommon.imageHandling import readNifti

# Setup image path list
IMAGE_FMT_STRING = "image-{}-deriv.nii"
DERIV_DIR = os.path.join("images", "deriv")
PATH_FMT_STRING = os.path.join(DERIV_DIR, IMAGE_FMT_STRING)

paths = [PATH_FMT_STRING.format(i) for i in range(1, 6)]

# Read in NIfTI images to memory
images = [readNifti(path) for path in paths]

# Setup list of BIDS-I objects
testMetadata = {'subject': '01', 'task': 'test', 'suffix': 'bold', 'datatype': 'func', 
                'session': '01', 'run': 1, 'RepetitionTime': 1.5, 'EchoTime': 0.5}
incrementals = [BidsIncremental(image, testMetadata) for image in images]

# Setup array to hold the measurement data
NUM_SAMPLES = 1000 
NUM_EXPERIMENTS = 2  # serialize and deserialize
SERIALIZE_IDX = 0
DESERIALIZE_IDX = 1
shape = (len(incrementals), 2, NUM_SAMPLES)
measurement_data = np.zeros(shape, dtype=np.float64)
print("shape:", measurement_data.shape)

# Start loop for each BIDS-I object, 
for idx, incremental in enumerate(incrementals):
    # Start loop for N iterations of: pickle the object, recording the time taken, then unpickle the object, recording the time taken
    startTime = 0.0
    for i in range(NUM_SAMPLES):
        startTime = time.process_time()
        pickled = pickle.dumps(incremental)
        serializationTime = time.process_time() - startTime
        measurement_data[idx][SERIALIZE_IDX][i] = serializationTime

        startTime = time.process_time()
        unpickled = pickle.loads(pickled)
        deserializationTime = time.process_time() - startTime
        measurement_data[idx][DESERIALIZE_IDX][i] = deserializationTime

        assert unpickled == incremental


# Reduce the 3-D matrix to a 2-D matrix of average time to pickle each object and average time to unpickle each object
averages = np.average(measurement_data, axis=2)

# Output the 2-D matrix as a CSV file for chart generation in Excel, as matplotlib is fiddly
averages *= 1000
np.savetxt('dataout-ms.txt', averages, delimiter=',', header='Serialize Average Time (ms), Deserialize Average Time (ms)')
