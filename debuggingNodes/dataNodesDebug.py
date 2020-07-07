from dataNodes.dataLoaderNode import DatasetLoader, CollateOptions
from dataNodes.edaNode import EdaManager
from dataNodes.dataAugmentationNode import DatasetAugmentation
from executionNodes.loggerNode import logs
from executionNodes.settings import SettingsManager


class DataNodesDebugPackage():
    def __init__(self):
        """
        constructor
        :attr settings [dict] : dictionary of user defined settings for Speedpack
        """
        try:
            sm = SettingsManager()
            self.dataLoaderSettings = sm.readField("dataLoaderNode")
            logs.debugging.info("DEBUG: DatasetLoaderNode settings loaded successfully")
        except Exception as e:
            logs.debugging.error("DEBUG: Error while opening settings file, %s", e)

    def testVanillaDatasetLoader(self):
        """
        test the default dataset loader using the defined dataset directory
        :return result [bool] : result of whether the test passed or failed
        """
        result = True
        try:
            datasetLoader = DatasetLoader(self.dataLoaderSettings["dataDir"],
                                          collateFn=CollateOptions.generalCollate)
            logs.debugging.info("testVanillaDatasetLoader: DatasetLoader creation successful")
        except Exception as e:
            result = False
            logs.debugging.error("testVanillaDatasetLoader: DatasetLoader creation unsuccessful: %s", e)
        # debug the training data loader
        try:
            for i, traindata in enumerate(datasetLoader.trainDeviceDataLoader):
                images, labels = traindata
            logs.debugging.info("testVanillaDatasetLoader: trainDeviceDataLoader iteration successful")
        except Exception as e:
            result = False
            logs.debugging.error("testVanillaDatasetLoader: trainDeviceDataLoader unsuccessful: %s", e)
        # debug the testing data loader
        try:
            for i, testdata in enumerate(datasetLoader.testDeviceDataLoader):
                images, labels = testdata
            logs.debugging.info("testVanillaDatasetLoader: testDeviceDataLoader iteration successful")
        except Exception as e:
            result = False
            logs.debugging.error("testVanillaDatasetLoader: testDeviceDataLoader unsuccessful: %s", e)
        return result

    def testAugmentedDatasetLoader(self):
        """
        test the dataset loader using the defined dataset directory and with data augmentation activities
        :return result [bool] : result of whether the test passed or failed
        """
        result = True
        try:
            da = DatasetAugmentation()
            datasetLoader = DatasetLoader(self.dataLoaderSettings["dataDir"],
                                          collateFn=CollateOptions.generalCollate,
                                          dataAugmentationObj=da.transforms)
            logs.debugging.info("testAugmentedDatasetLoader: DatasetLoader creation successful")
        except Exception as e:
            result = False
            logs.debugging.error("testAugmentedDatasetLoader: DatasetLoader creation unsuccessful: %s", e)
        # debug the training data loader
        try:
            for i, traindata in enumerate(datasetLoader.trainDeviceDataLoader):
                images, labels = traindata
            logs.debugging.info("testAugmentedDatasetLoader: trainDeviceDataLoader iteration successful")
        except Exception as e:
            result = False
            logs.debugging.error("testAugmentedDatasetLoader: trainDeviceDataLoader unsuccessful: %s", e)
        # debug the testing data loader
        try:
            for i, testdata in enumerate(datasetLoader.testDeviceDataLoader):
                images, labels = testdata
            logs.debugging.info("testAugmentedDatasetLoader: testDeviceDataLoader iteration successful")
        except Exception as e:
            result = False
            logs.debugging.error("testAugmentedDatasetLoader: testDeviceDataLoader unsuccessful: %s", e)
        return result

    def testVanillaEdaManager(self):
        """
        test the default EDA manager using the defined dataset directory
        :return result [bool] : result of whether the test passed or failed
        """
        result = True
        try:
            datasetLoader = DatasetLoader(self.dataLoaderSettings["dataDir"],
                                          collateFn=CollateOptions.generalCollate)
            edaManager = EdaManager(datasetLoader)
            logs.debugging.info("testVanillaEdaManager: EdaManager creation successful")
        except Exception as e:
            result = False
            logs.debugging.error("testVanillaEdaManager: EdaManager creation unsuccessful: %s", e)
        # debug the randomExamples function
        try:
            edaManager.rawExamples([3, 3], "Vanilla")
            logs.debugging.info("testVanillaEdaManager: rawExamples function successful")
        except Exception as e:
            result = False
            logs.debugging.error("testVanillaEdaManager: rawExamples function unsuccessful: %s", e)
        return result

    def testAugmentedEdaManager(self):
        """
        test the EDA manager with data augmentation activity using the defined dataset directory
        :return result [bool] : result of whether the test passed or failed
        """
        result = True
        try:
            da = DatasetAugmentation()
            datasetLoader = DatasetLoader(self.dataLoaderSettings["dataDir"],
                                          collateFn=CollateOptions.generalCollate,
                                          dataAugmentationObj=da.transforms)
            edaManager = EdaManager(datasetLoader)
            logs.debugging.info("testAugmentedEdaManager: EdaManager creation successful")
        except Exception as e:
            result = False
            logs.debugging.error("testAugmentedEdaManager: EdaManager creation unsuccessful: %s", e)
        # debug the randomExamples function
        try:
            edaManager.rawExamples([3, 3], "Augmented")
            logs.debugging.info("testAugmentedEdaManager: rawExamples function successful")
        except Exception as e:
            result = False
            logs.debugging.error("testAugmentedEdaManager: rawExamples function unsuccessful: %s", e)
        return result


def run():
    """
    run the DataNodesDebugPackage and log errors/successes
    :return result [bool] : result of whether the test passed or failed
    """
    result = True
    debugPackage = DataNodesDebugPackage()
    # result = debugPackage.testVanillaDatasetLoader() and result
    # result = debugPackage.testAugmentedDatasetLoader() and result
    # result = debugPackage.testVanillaEdaManager() and result
    result = debugPackage.testAugmentedEdaManager() and result
    logs.debugging.info("DEBUG: DataNodesDebug finished running")
    return result
