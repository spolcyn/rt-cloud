"""
Tests the timing of appending as function of image and metadata size
"""

import gc
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from tqdm import tqdm
from bids.exceptions import NoMatchError

rtCloudDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(rtCloudDir)
sys.path.append(rtCloudDir)
from rtCommon.bidsIncremental import BidsIncremental
from rtCommon.bidsArchive import BidsArchive
from rtCommon.bidsRun import BidsRun

logger = logging.getLogger(__name__)

DATASET_DIR = 'datasets'
DATASET_DIR_FMT = os.path.join(DATASET_DIR, '{}-download')
TARGET_DIR = 'tmp_out'
tmpdir = tempfile.gettempdir()
print("Temp dir:", tmpdir)

# DATASET_NUMBERS = ['ds002551', 'ds003090', 'ds000138', 'ds002750', 'ds002733']
DATASET_NUMBERS = ['ds003090']
# others:
# ds003440: 815.49MB

TESTING_NEW = False
TESTING_OLD = not TESTING_NEW

def prod(t1):
    # Returns product of tuple elements
    prd = 1
    for el in t1:
        prd *= el
    return prd

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
incrementals = []

# For each dataset;
for dataset_idx, dataset_num in enumerate(DATASET_NUMBERS):
    print("Running dataset", dataset_num)
    # Load up the archive
    archive = BidsArchive(DATASET_DIR_FMT.format(dataset_num))

    # Setup the metadata to get all the incrementals from the archive
    subjects = archive.getSubjects()
    tasks = archive.getTasks()
    sessions = archive.getSessions()
    runs = archive.getRuns()

    # Some lists may be empty, so make them have at least 'None' in them so
    # they still can be iterated over and the inner run loop can be executed
    for maybe_empty_list in [runs, sessions]:
        if len(maybe_empty_list) == 0:
            maybe_empty_list.append(None)

    print('subjects:', subjects)
    print('tasks:', tasks)
    print('sessions:', sessions)
    print('runs:', runs)

    currentImageShape = None

    # Get all the incrementals into a list
    get_times = [] 
    bids_runs = []

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

                    # All images should have the same number of voxels, but
                    # confirm that here just to be sure by setting the image
                    # shape for the first image, and making sure all future
                    # images have the same number of voxels. 4th dimensions
                    # (time) may differ, so don't check that, and some studies
                    # seem to swap dimensions some times (e.g., sometimes they
                    # have (50, 40, 30, 20) and other images are (50, 30, 40,
                    # 20). Since all we're concerned about is the voxel size
                    # for the images in each run, this is ok.  See OpenNeuro
                    # DS003090 as an example -- sub-2006 and sub-2011 have a
                    # dimension inverted.
                    if currentImageShape is None:
                        currentImageShape = image.shape
                        shape_dict[dataset_num] = currentImageShape
                    else:
                        prod(image.shape[:3]) == prod(currentImageShape[:3]), "Image.shape: {}, Current: {}".format(image.shape, currentImageShape)

                    current_run = None
                    # Loop over all images in the volume until all possible incrementals are extracted
                    for i in tqdm(range(images_in_volume), "Images in volume", position=4, leave=False):
                        # Start the timer
                        startTime = time.process_time()
                        if TESTING_NEW:
                            if current_run is None:
                                current_run = archive.getBidsRun(**entities)
                            # Get
                            incremental = current_run.getIncremental(i)
                        elif TESTING_OLD:
                            incremental = archive._getIncremental(**entities)
                        else:
                            assert False
                        # Store get time in measurement data
                        timeTaken = time.process_time() - startTime
                        get_times.append(timeTaken)
                        if TESTING_OLD:
                            incrementals.append(incremental)

                    if TESTING_NEW:
                        bids_runs.append(current_run)


    # Create the path to the new archive
    new_archive_path = os.path.join(tmpdir, "2-q-eval", "{}-new".format(dataset_num))
    shutil.rmtree(new_archive_path, ignore_errors=True)
    new_archive = BidsArchive(new_archive_path)
    append_times = []

    print("Opened archive", new_archive_path, "for reading")

    # Loop over all incrementals until the archive is fully remade
    append_successful = False

    if TESTING_NEW:
        for bids_run in tqdm(bids_runs, desc="BIDS Runs", position=1):
            new_run = BidsRun()
            for i in tqdm(range(bids_run.numIncrementals()), desc="Incrementals", leave=False):
                incremental = bids_run.getIncremental(i)
                # Start the timer
                startTime = time.process_time()
                # Append
                new_run.appendIncremental(incremental)
                # Store append time in measurement data
                timeTaken = time.process_time() - startTime
                
                # If it's the last incremental, also append to the archive
                if (i + 1) == bids_run.numIncrementals():
                    startTime = time.process_time()
                    new_archive.appendBidsRun(new_run)
                    timeTaken = time.process_time() - startTime + timeTaken

                append_times.append(timeTaken)

    elif TESTING_OLD:
        for incremental in tqdm(incrementals, desc="Incrementals", position=1):
            # Start the timer
            startTime = time.process_time()
            # Append
            new_archive._appendIncremental(incremental)
            # Store append time in measurement data
            timeTaken = time.process_time() - startTime
            append_times.append(timeTaken)


    # Store the data for later processing
    op_data_dict = {'get': get_times, 'append': append_times}
    measurement_data[dataset_num] = op_data_dict
    incrementals = []

    gc.collect()

    # Verify correctness
    """
    print("Verifying correctness")
    for subject in tqdm(subjects, "Subjects", position=0):
        for task in tqdm(tasks, "Tasks", position=1, leave=False):
            for session in tqdm(sessions, "Sessions", position=2, leave=False):
                for run in tqdm(runs, "Runs", position=3, leave=False):
                    entities = {'subject': subject, 'task': task, 'session': session, 'run': run, 
                                'datatype': 'func'}
                    # filter out the None entities
                    entities = {e: entities[e] for e in entities if entities[e] is not None}

                    try:
                        run1 = archive.getBidsRun(**entities)
                        run2 = new_archive.getBidsRun(**entities)
                        # assert run1 == run2
                    except NoMatchError:
                        continue  # skip
    """


# Aggregate, summarize, and output get and append data
stats_dict = {'stddev': np.std, 'mean': np.mean, 'min': np.min, 'max': np.max, 'sum': np.sum}

new_old_string = "new" if TESTING_NEW else "old"

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
        output_path = os.path.join("get-append-eval", "{}-eval-{}-{}.txt".format(
            dataset_num, op_name, new_old_string))
        np.savetxt(output_path, data_array, header=header_string)

