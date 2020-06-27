from debuggingNodes import dataNodesDebug
from executionNodes.loggerNode import logs

if __name__ == '__main__':
    if dataNodesDebug.run():
        logs.debugging.info("dataNodesDebug passed all tests")
    else:
        logs.debugging.error("dataNodesDebug DID NOT pass all tests")
