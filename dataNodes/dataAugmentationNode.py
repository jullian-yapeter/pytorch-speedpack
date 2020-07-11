from executionNodes.loggerNode import logs
from executionNodes.settings import SettingsManager
import torchvision.transforms as tt


class DatasetAugmentation():
    def __init__(self):
        """
        constructor
        create a PyTorch transforms for both the training and testing phase in accordance with defined settings
        :attr dataAugmentationSettings [dict] : user-chosen options of augmentation to be applied
        :attr transforms [dict] : dictionary of transforms (for training and test sets)
        """
        try:
            sm = SettingsManager()
            self.dataAugmentationSettings = sm.readField("dataAugmentationNode")
            logs.dataAugmentation.info("data augmentation settings loaded successfully")
        except Exception as e:
            logs.dataAugmentation.error("Error while opening settings file, %s", e)
        self.transforms = self.composeTransform()

    def composeTransform(self):
        """
        builds a PyTorch transform based on the passed in settings
        :return transforms [dict] : dictionary of transforms (for training and test sets)
        """
        try:
            transforms = {'train': tt.ToTensor(), 'test': tt.ToTensor()}  # default
            trainTransforms = []
            testTransforms = []
            # construct training transforms
            trainSettings = self.dataAugmentationSettings["train"]
            trainAugmentationSelections = trainSettings.keys()
            if "resize" in trainAugmentationSelections:
                trainTransforms.append(tt.Resize(trainSettings.get("resize")))
            if "hflip" in trainAugmentationSelections:
                trainTransforms.append(tt.RandomHorizontalFlip(trainSettings.get("hflip")))
            if "vflip" in trainAugmentationSelections:
                trainTransforms.append(tt.RandomVerticalFlip(trainSettings.get("vflip")))
            trainTransforms.append(tt.ToTensor())
            transforms["train"] = tt.Compose(trainTransforms)
            # construct testing transforms
            testSettings = self.dataAugmentationSettings["test"]
            testAugmentationSelections = testSettings.keys()
            if "resize" in testAugmentationSelections:
                testTransforms.append(tt.Resize(testSettings.get("resize")))
            if "hflip" in testAugmentationSelections:
                testTransforms.append(tt.RandomHorizontalFlip(testSettings.get("hflip")))
            if "vflip" in testAugmentationSelections:
                testTransforms.append(tt.RandomVerticalFlip(testSettings.get("vflip")))
            testTransforms.append(tt.ToTensor())
            transforms["test"] = tt.Compose(testTransforms)
            logs.dataAugmentation.info("data augmentation transforms composed successfully")
            return transforms
        except Exception as e:
            logs.dataAugmentation.error("Error data augmentation transforms composition failed, %s", e)
