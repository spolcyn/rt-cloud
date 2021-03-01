# import important modules
import os
import sys
import numpy
import argparse

currPath = os.path.dirname(os.path.realpath(__file__))
rootPath = os.path.dirname(os.path.dirname(currPath))
sys.path.append(rootPath)

# import project modules from rt-cloud
from rtCommon.utils import loadConfigFile, stringPartialFormat
from rtCommon.clientInterface import ClientInterface

# path for default configuration toml file
defaultConfig = os.path.join(currPath, 'conf/openNeuroClient.toml')


def doRuns(cfg, bidsInterface, subjInterface, webInterface):
    run = cfg.runNum[0]
    subject = cfg.subjectName
    entities = {'subject': subject, 'run': run, 'suffix': 'bold', 'datatype': 'func'}
    streamId = bidsInterface.initOpenNeuroStream(cfg.dsAccessionNumber, **entities)
    numVols = bidsInterface.getNumVolumes(streamId)
    webInterface.clearRunPlot(run)
    for idx in range(numVols):
        bidsIncremental = bidsInterface.getIncremental(streamId, idx)
        imageData = bidsIncremental.imageData
        avg_niftiData = numpy.mean(imageData)
        print("| average activation value for TR %d is %f" %(idx, avg_niftiData))
        webInterface.plotDataPoint(run, idx, float(avg_niftiData))


def main(argv=None):
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--config', '-c', default=defaultConfig, type=str,
                           help='experiment config file (.json or .toml)')
    argParser.add_argument('--runs', '-r', default=None, type=str,
                           help='Comma separated list of run numbers')
    argParser.add_argument('--yesToPrompts', '-y', default=False, action='store_true',
                           help='automatically answer tyes to any prompts')
    args = argParser.parse_args(argv)

    # Initialize the RPC connection to the projectInterface
    # This will give us a dataInterface for retrieving files and
    # a subjectInterface for giving feedback
    clientInterfaces = ClientInterface(yesToPrompts=args.yesToPrompts)
    bidsInterface = clientInterfaces.bidsInterface
    subjInterface = clientInterfaces.subjInterface
    webInterface  = clientInterfaces.webInterface

    # load the experiment configuration file
    cfg = loadConfigFile(args.config)
    doRuns(cfg, bidsInterface, subjInterface, webInterface)
    return 0


if __name__ == "__main__":
    main(sys.argv)
    sys.exit(0)
