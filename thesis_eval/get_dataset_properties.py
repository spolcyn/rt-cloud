import os
import sys

import numpy as np
import pandas as pd

rtCloudDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(rtCloudDir)
sys.path.append(rtCloudDir)
from rtCommon.bidsArchive import BidsArchive

DATASET_NUMBERS = ['ds002551', 'ds003440', 'ds000138', 'ds002750', 'ds002733']
DATASET_DIR = 'datasets'
DATASET_DIR_FMT = os.path.join(DATASET_DIR, '{}-download')


small_large_by_dataset = {key: {} for key in DATASET_NUMBERS}
# For each dataset;
for dataset_idx, dataset_num in enumerate(DATASET_NUMBERS):
    print("Running dataset", dataset_num)
    # Load up the archive
    archive = BidsArchive(DATASET_DIR_FMT.format(dataset_num))

    smallest = None
    largest = None
    dtype = None
    for img in archive.getImages(datatype='func'):
        img = img.get_image()
        dtype = img.header['datatype']
        if smallest is None:
            smallest = img.shape
        if largest is None:
            largest = img.shape

        if np.prod(smallest) > np.prod(img.shape):
            smallest = img.shape
        if np.prod(largest) < np.prod(img.shape):
            largest = img.shape

    print('Dataset: ', dataset_num, ' | Total voxels:', np.prod(largest))

    small_large_by_dataset[dataset_num] = {'Smallest Image in Dataset':
                                           smallest, 'Largest Image in Dataset':
                                           largest, 'Datatype': dtype}

print(small_large_by_dataset)
df = pd.DataFrame.from_dict(data=small_large_by_dataset,
        orient='index')#, columns=columns)
df.index.name = 'Dataset ID Number'
df.to_csv('dataset-range.tsv', sep='\t')
print(df)


