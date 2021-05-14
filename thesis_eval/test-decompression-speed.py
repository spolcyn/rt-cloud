import time
import gzip
import subprocess
from shutil import copyfile
import shutil
import os
from tqdm import tqdm

totalTime = 0
ITERATIONS = 5

SOURCE_PATH = '/tmp/test_source.nii.gz'
TMP_PATH = '/tmp/file.nii.gz'
TMP_PATH_DECOMP = '/tmp/file.nii'

def useSubprocess(source, dest):
    subprocess.run(['gunzip', source])

def useGzipCopyFileObj(source, dest):
    with open(dest, 'wb') as f_out:
        with gzip.open(source, 'rb') as f_in:
            shutil.copyfileobj(f_in, f_out)

def useGzipDecompress(source, dest):
    with open(source, 'rb') as f_in:
        with open(dest, 'wb') as f_out:
            f_out.write(gzip.decompress(f_in.read()))

# https://stackoverflow.com/questions/43482006/faster-better-gunzip-and-general-file-input-output-in-python
def useGzipBuffer(source, dest):
    buf = bytearray(8192)
    with open(dest, 'wb') as fout:
        with gzip.open(source, 'rb') as fin:
            while fin.readinto(buf):
                fout.write(buf)

for method in tqdm([useSubprocess, useGzipCopyFileObj, useGzipDecompress,
                    useGzipBuffer], position=0):
    totalTime = 0
    for i in tqdm(range(ITERATIONS), position=1):
        try:
            os.remove(TMP_PATH_DECOMP)
        except FileNotFoundError:
            pass
        copyfile(SOURCE_PATH, TMP_PATH)
        startTime = time.time()
        method(TMP_PATH, TMP_PATH_DECOMP)
        totalTime += (time.time() - startTime)

    print('\nAverage for', method.__name__, ':', totalTime/ITERATIONS, '\n')

