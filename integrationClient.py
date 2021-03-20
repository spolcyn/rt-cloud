import time

from rtCommon.clientInterface import ClientInterface

entities = {'subject': '01', 'run': 1}
clientInterface = ClientInterface()
streamId = clientInterface.bidsInterface.initOpenNeuroStream("ds002014", **entities)

numVolumes = clientInterface.bidsInterface.getNumVolumes(streamId)

print("NumVolumes:", numVolumes)

start = time.process_time()

for i in range(numVolumes):
    clientInterface.bidsInterface.getIncremental(streamId)

end = time.process_time()

print("Elapsed:", end - start)
