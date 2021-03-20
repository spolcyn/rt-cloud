"""
Tests the timing of appending as function of image and metadata size
"""

import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from tqdm import tqdm

rtCloudDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(rtCloudDir)
sys.path.append(rtCloudDir)
from rtCommon.bidsIncremental import BidsIncremental
from rtCommon.bidsArchive import BidsArchive

DATASET_DIR = 'datasets'
DATASET_DIR_FMT = os.path.join(DATASET_DIR, '{}-download')
TARGET_DIR = 'tmp_out'
tmpdir = tempfile.gettempdir()

DATASET_NUMBERS = ['ds000138', 'ds003090', 'ds002750', 'ds002733', 'ds002551']
# others:
# ds003440: 815.49MB

# download datasets
for dataset_num in DATASET_NUMBERS:
    dataset_path = DATASET_DIR_FMT.format(dataset_num)

    if os.path.exists(dataset_path):
        print("Already have dataset", str(dataset_num))
        continue
    
    command = 'aws s3 sync --no-sign-request s3://openneuro.org/{num} {path}/'.format(
                num=dataset_num, path=dataset_path)
    command = command.split(' ')
    print("Downloading", dataset_num)
    assert subprocess.call(command, stdout=subprocess.DEVNULL) == 0, "S3 download failed"

    print("Gunzipping", dataset_num)
    command = ['gunzip', '-r', dataset_path]
    assert subprocess.call(command) == 0, "Gunzip failed"

    print("Finished downloading and gunzipping dataset", dataset_num)


shape = (len(DATASET_NUMBERS), 2)
SUM_INDEX = 0
COUNT_INDEX = 1
measurement_data = {}
shape_dict = {}

# For each dataset;
for dataset_idx, dataset_num in enumerate(DATASET_NUMBERS):
    print("Running dataset", dataset_num)
    # Load up the archive
    archive = BidsArchive(DATASET_DIR_FMT.format(dataset_num))

    # Setup the metadata to get all the incrementals from the archive
    runs = archive.getRuns()
    tasks = archive.getTasks()
    subjects = archive.getSubjects()
    sessions = archive.getSessions()

    print('subjects:', subjects)
    print('tasks:', tasks)
    print('sessions:', sessions)
    print('runs:', runs)

    for maybe_empty_list in [runs, sessions]:
        if len(maybe_empty_list) == 0:
            maybe_empty_list.append(None)

    currentImageShape = None

    # Get all the incrementals into a list
    incrementals = []
    get_times = [] 

    for subject in tqdm(subjects, "Subjects", position=0):
        for task in tqdm(tasks, "Tasks", position=1, leave=False):
            for session in tqdm(sessions, "Sessions", position=2, leave=False):
                for run in tqdm(runs, "Runs", position=3, leave=False):
                    entities = {'subject': subject, 'task': task, 'session': session, 'run': run, 
                                'datatype': 'func'}
                    # filter out the None entities
                    entities = {e: entities[e] for e in entities if entities[e] is not None}

                    images = archive.getImages(**entities)
                    # possible that not all subjects have all tasks, sessions,
                    # or runs, so if can't find particular combo, just continue
                    if len(images) == 0:  
                        continue
                    elif len(images) > 1:
                        assert False, "Got more than one image from search"
                    image = images[0].get_image()
                    images_in_volume = image.shape[3]

                    # All images should have the same 3-D dimensions, but
                    # confirm that here just to be sure by setting the image
                    # shape for the first image, and making sure all future
                    # images match that shape. 4th dimensions (time) may
                    # differ, so don't check that.
                    if currentImageShape is None:
                        currentImageShape = image.shape
                        shape_dict[dataset_num] = currentImageShape
                    else:
                        assert image.shape[:3] == currentImageShape[:3], "Image.shape: {}, Current: {}".format(image.shape, currentImageShape)

                    # Loop over all images in the volume until all possible incrementals are extracted
                    for i in range(images_in_volume):
                        # Start the timer
                        startTime = time.process_time()
                        # Get
                        incremental = archive.getIncremental(imageIndex=i, **entities)
                        # Store get time in measurement data
                        timeTaken = time.process_time() - startTime
                        get_times.append(timeTaken)
                        incrementals.append(incremental)

    # Create the path to the new archive
    new_archive_path = os.path.join(tmpdir, "2-q-eval", "{}-new".format(dataset_num))
    shutil.rmtree(new_archive_path, ignore_errors=True)
    new_archive = BidsArchive(new_archive_path)
    append_times = []

    # Loop over all incrementals until the archive is fully remade
    append_successful = False
    for incremental in tqdm(incrementals, desc="Incrementals"):
        # Start the timer
        startTime = time.process_time()
        # Append
        append_successful = new_archive.appendIncremental(incremental)
        # Store append time in measurement data
        timeTaken = time.process_time() - startTime
        assert append_successful
        append_times.append(timeTaken)

    # Store the data for later processing
    op_data_dict = {'get': get_times, 'append': append_times}
    measurement_data[dataset_num] = op_data_dict


# Aggregate, summarize, and output get and append data
stats_dict = {'stddev': np.std, 'mean': np.mean, 'min': np.min, 'max': np.max, 'sum': np.sum}

for dataset_num, list_dict in measurement_data.items():
    for op_name, data_list in list_dict.items():
        data_array = np.array(data_list)

        desc_string = 'Data for Dataset: {}\nImage Size: {}\nMeasurement: Runtime for each {} (sec)\n\n'.format(dataset_num, shape_dict[dataset_num], op_name)
        stat_string = '-----Stats for {}-----\n'.format(op_name)
        for stat_name, fx in stats_dict.items():
            stat_string += '{name}: {value}\n'.format(name=stat_name, value=fx(data_array))
        print(stat_string)

        header_string = desc_string + stat_string
        # Output data
        output_path = os.path.join("get-append-eval", "{}-eval-{}.txt".format(dataset_num, op_name))
        np.savetxt(output_path, data_array, header=header_string)

