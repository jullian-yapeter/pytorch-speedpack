from executionNodes.loggerNode import logs
from torchvision import transforms
import json


class DatasetAugmentation():
    def __init__(self):
        self.transforms = {'train': transforms.ToTensor(), 'test': transforms.ToTensor()}  # default
        try:
            with open('settings.json') as settingsFile:
                self.settings = json.load(settingsFile)["dataAugmentationNode"]
        except Exception as e:
            logs.debugging.error("Error while opening settings file, %s", e)

    def composeTransform(self):
        pass
