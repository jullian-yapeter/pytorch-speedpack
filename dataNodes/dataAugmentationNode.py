from executionNodes.loggerNode import logs
import torchvision.transforms as tt
import json


class DatasetAugmentation():
    def __init__(self):
        """
        constructor
        create a PyTorch transforms for both the training and testing phase in accordance with defined settings
        """
        self.transforms = self.composeTransforms
        try:
            with open('settings.json') as settingsFile:
                self.settings = json.load(settingsFile)["dataAugmentationNode"]
            logs.debugging.info("data augmentation settings loaded successfully")
        except Exception as e:
            logs.debugging.error("Error while opening settings file, %s", e)

    def composeTransform(self):
        transforms = {'train': tt.ToTensor(), 'test': tt.ToTensor()}  # default
        trainTransforms = []
        testTransforms = []
        # construct training transforms
        if "hflip" in self.settings["train"]:
            trainTransforms.append(tt.RandomHorizontalFlip())
        if "vflip" in self.settings["train"]:
            trainTransforms.append(tt.RandomVerticalFlip())
        trainTransforms.append(tt.ToTensor())
        transforms["train"] = tt.Compose(trainTransforms)
        # construct testing transforms
        if "hflip" in self.settings["test"]:
            testTransforms.append(tt.RandomHorizontalFlip())
        if "vflip" in self.settings["test"]:
            testTransforms.append(tt.RandomVerticalFlip())
        testTransforms.append(tt.ToTensor())
        transforms["test"] = tt.Compose(testTransforms)
        return transforms
