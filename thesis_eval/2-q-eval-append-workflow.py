"""
Tests the timing of appending as function of image and metadata size

Outputs data based on the workflow
"""

import json
import logging
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time

from tqdm import tqdm

rtCloudDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(rtCloudDir)
sys.path.append(rtCloudDir)
from rtCommon.bidsArchive import BidsArchive # noqa
from rtCommon.bidsRun import BidsRun # noqa

logger = logging.getLogger(__name__)

DATASET_DIR = 'datasets'
DATASET_DIR_FMT = os.path.join(DATASET_DIR, '{}-download')
TARGET_DIR = 'tmp_out'
tmpdir = tempfile.gettempdir()
print("Temp dir:", tmpdir)

DATASET_NUMBERS = ['ds002551', 'ds003440', 'ds000138', 'ds002750', 'ds002733']
# DATASET_NUMBERS = ['ds002551']
# DATASET_NUMBERS = ['ds003440']
# DATASET_NUMBERS = ['ds002750']


def prod(t1):
    # Returns product of tuple elements
    prd = 1
    for el in t1:
        prd *= el
    return prd


def download_and_unzip_datasets(dataset_numbers):
    for dataset_num in dataset_numbers:
        dataset_path = DATASET_DIR_FMT.format(dataset_num)

        if os.path.exists(dataset_path):
            print("Already have dataset", str(dataset_num))
            continue

        command = ('aws s3 sync --no-sign-request '
                   's3://openneuro.org/{num} {path}/'
                   .format(num=dataset_num, path=dataset_path))
        command = command.split(' ')
        print("Downloading", dataset_num)
        assert subprocess.call(command, stdout=subprocess.DEVNULL) == 0, \
            "S3 download failed"

        print("Gunzipping", dataset_num)
        # command = ['gunzip', '-r', dataset_path]
        # assert subprocess.call(command) == 0, "Gunzip failed"

        print("Finished downloading and gunzipping dataset", dataset_num)


download_and_unzip_datasets(DATASET_NUMBERS)

measurement_data = {}
shape_dict = {}
incrementals = []


def run_archive_loop(dataset_num, per_run_fn):
    """
    Executes provided per_run_fn for every run of the archive
    """
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

    # Create the path to the new archive
    new_archive_path = os.path.join(tmpdir, "2-q-eval",
                                    "{}-new".format(dataset_num))
    shutil.rmtree(new_archive_path, ignore_errors=True)
    new_archive = BidsArchive(new_archive_path)

    for subject in tqdm(subjects, "Subjects", position=0):
        for task in tqdm(tasks, "Tasks", position=1, leave=False):
            for session in tqdm(sessions, "Sessions", position=2, leave=False):
                for run in tqdm(runs, "Runs", position=3, leave=False):
                    entities = {'subject': subject, 'task': task, 'session':
                                session, 'run': run, 'datatype': 'func'}
                    # filter out the None entities
                    entities = {e: entities[e] for e in entities if entities[e]
                                is not None}

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
                        prod(image.shape[:3]) == prod(currentImageShape[:3]),
                        "Image.shape: {}, Current: {}".format(image.shape,
                                                              currentImageShape)

                    sidecar_metadata = archive.getSidecarMetadata(images[0])

                    per_run_fn(image=image, images_in_volume=images_in_volume,
                               sidecar_metadata=sidecar_metadata,
                               entities=entities,
                               archive=archive, new_archive=new_archive,
                               subject=subject, task=task, session=session,
                               run=run)


def analyze_workflow1(dataset_num):
    # Workflow 1: MRI Machine -> Incremental
    print("Running workflow1 for dataset", dataset_num)

    # These are the operations we'll store data for
    keys = ['serialize', 'deserialize']
    workflow1_data_dict = {key: [] for key in keys}

    def per_run_fn(images_in_volume, new_archive, archive, entities, **kwargs):
        # The BIDS Run we're getting data from, simulating the MRI machine
        data_source_run = archive.getBidsRun(**entities)

        # Loop over all incrementals in the run
        for i in tqdm(range(images_in_volume), "Images in volume", position=4,
                      leave=False):
            # Get the incremental
            incremental = data_source_run.getIncremental(i)

            # Do serialization/deserialization round trip
            startTime = time.process_time()
            serialized_incremental = pickle.dumps(incremental)
            timeTaken = time.process_time() - startTime
            workflow1_data_dict['serialize'].append(timeTaken)

            startTime = time.process_time()
            incremental = pickle.loads(serialized_incremental)
            timeTaken = time.process_time() - startTime
            workflow1_data_dict['deserialize'].append(timeTaken)

    run_archive_loop(dataset_num, per_run_fn)

    # Now return the dictionary for json dumping
    return workflow1_data_dict


def run_and_write_workflow1():
    for dataset_number in DATASET_NUMBERS:
        result = analyze_workflow1(dataset_number)
        path = os.path.join('workflow1', dataset_number + '-workflow1.txt')
        with open(path, 'w') as f:
            json.dump(result, f)


run_and_write_workflow1()


def analyze_workflow2(dataset_num):
    # Workflow 2: MRI Machine -> Incremental -> Run -> Archive
    print("Running workflow2 for dataset", dataset_num)

    # These are the operations we'll store data for
    keys = ['serialize', 'deserialize', 'appendIncremental', 'appendBidsRun']
    workflow2_data_dict = {key: [] for key in keys}

    def per_run_fn(images_in_volume, new_archive, archive, entities, **kwargs):
        # The working BIDS Run we'll add to
        current_run = BidsRun()

        # The BIDS Run we're getting data from, simulating the MRI machine
        data_source_run = archive.getBidsRun(**entities)

        # Loop over all incrementals in the run
        for i in tqdm(range(images_in_volume), "Images in volume", position=4,
                      leave=False):
            # Get the incremental
            incremental = data_source_run.getIncremental(i)

            # Do serialization/deserialization round trip
            startTime = time.process_time()
            serialized_incremental = pickle.dumps(incremental)
            timeTaken = time.process_time() - startTime
            workflow2_data_dict['serialize'].append(timeTaken)

            startTime = time.process_time()
            incremental = pickle.loads(serialized_incremental)
            timeTaken = time.process_time() - startTime
            workflow2_data_dict['deserialize'].append(timeTaken)

            # Do appendIncremental to BIDS Run
            startTime = time.process_time()
            current_run.appendIncremental(incremental)
            timeTaken = time.process_time() - startTime
            workflow2_data_dict['appendIncremental'].append(timeTaken)

        # Do appendBidsRun to BIDS Archive
        startTime = time.process_time()
        new_archive.appendBidsRun(current_run)
        timeTaken = time.process_time() - startTime
        workflow2_data_dict['appendBidsRun'].append(timeTaken)

    run_archive_loop(dataset_num, per_run_fn)

    # Now return the dictionary for json dumping
    return workflow2_data_dict


def run_and_write_workflow2():
    for dataset_number in DATASET_NUMBERS:
        result = analyze_workflow2(dataset_number)
        path = os.path.join('workflow2', dataset_number + '-workflow2.txt'),
        with open(path, 'w') as f:
            json.dump(result, f)


run_and_write_workflow2()


def analyze_workflow3(dataset_num):
    # Workflow 3: BIDS Archive -> BIDS Run -> BIDS Incremental -> Analysis Model
    print("Running workflow3 for dataset", dataset_num)

    # These are the operations we'll store data for
    keys = ['getBidsRun', 'getIncremental', 'serialize', 'deserialize']
    workflow3_data_dict = {key: [] for key in keys}

    def per_run_fn(images_in_volume, archive, entities, **kwargs):
        # The working BIDS Run we'll get from
        startTime = time.process_time()
        current_run = archive.getBidsRun(**entities)
        timeTaken = time.process_time() - startTime
        workflow3_data_dict['getBidsRun'].append(timeTaken)

        # Loop over all incrementals in the run
        for i in tqdm(range(current_run.numIncrementals()), "Images in volume",
                      position=4, leave=False):
            # Get the incremental
            startTime = time.process_time()
            incremental = current_run.getIncremental(i)
            timeTaken = time.process_time() - startTime
            workflow3_data_dict['getIncremental'].append(timeTaken)

            # Do serialization/deserialization round trip
            startTime = time.process_time()
            serialized_incremental = pickle.dumps(incremental)
            timeTaken = time.process_time() - startTime
            workflow3_data_dict['serialize'].append(timeTaken)

            startTime = time.process_time()
            incremental = pickle.loads(serialized_incremental)
            timeTaken = time.process_time() - startTime
            workflow3_data_dict['deserialize'].append(timeTaken)

    run_archive_loop(dataset_num, per_run_fn)

    # Now return the dictionary for json dumping
    return workflow3_data_dict


for dataset_number in DATASET_NUMBERS:
    result = analyze_workflow3(dataset_number)
    path = os.path.join('workflow3', dataset_number + '-workflow3.txt')
    with open(path, 'w') as f:
        json.dump(result, f)
