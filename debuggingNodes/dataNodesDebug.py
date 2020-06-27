from dataNodes.dataLoaderNode import DatasetLoader
from executionNodes.loggerNode import logs
import json


class DataNodesDebugPackage():
    def __init__(self):
        with open('settings.json') as settingsFile:
            self.settings = json.load(settingsFile)["dataLoaderNode"]

    def testVanillaDatasetLoader(self):
        result = True
        try:
            _ = DatasetLoader(self.settings["dataDir"])
        except Exception as e:
            result = False
            logs.debugging.error("DEBUG: DatasetLoader creation unsuccessful: %s", e)
        return result


def run():
    result = True
    debugPackage = DataNodesDebugPackage()
    result = debugPackage.testVanillaDatasetLoader() and result
    return result
