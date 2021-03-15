import multiProcessing

import rtCommon.projectServer as ps
from rtCommon.structDict import StructDict
from tests.common import testPort

# from backgroundTestServers.py 
defaultCfg = StructDict({'sessionId': "test",
                         'subjectName': "test_sample",
                         'subjectNum': 1,
                         'subjectDay': 1,
                         'sessionNum': 1})

defaultProjectArgs = StructDict({'config': defaultCfg,
                                 'mainScript': 'projects/sample/sample.py',
                                 'port': testPort,
                                 'test': False})
args = defaultProjectArgs
args['dataRemote'] = True
args['subjectRemote'] = False


projectServer = ps.ProjectServer(args)
projectServer.start()
