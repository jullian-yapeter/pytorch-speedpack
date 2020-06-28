from dataNodes.dataLoaderNode import DatasetLoader
from executionNodes.loggerNode import logs
import json


class DataNodesDebugPackage():
    def __init__(self):
        """
        constructor
        :attr settings [dict] : dictionary of user defined settings for Speedpack
        """
        try:
            with open('settings.json') as settingsFile:
                self.settings = json.load(settingsFile)["dataLoaderNode"]
        except Exception as e:
            logs.debugging.error("Error while opening settings file, %s", e)

    def testVanillaDatasetLoader(self):
        """
        test the default dataset loader using the defined dataset directory
        :return result [bool] : result of whether the test passed or failed
        """
        result = True
        try:
            _ = DatasetLoader(self.settings["dataDir"])
        except Exception as e:
            result = False
            logs.debugging.error("DEBUG: DatasetLoader creation unsuccessful: %s", e)
        return result


def run():
    """
    run the DataNodesDebugPackage and log errors/successes
    :return result [bool] : result of whether the test passed or failed
    """
    result = True
    debugPackage = DataNodesDebugPackage()
    result = debugPackage.testVanillaDatasetLoader() and result
    return result
