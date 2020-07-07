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
            logs.debugging.info("data augmentation settings loaded successfully")
        except Exception as e:
            logs.debugging.error("Error while opening settings file, %s", e)
        self.transforms = self.composeTransform()

    def composeTransform(self):
        """
        builds a PyTorch transform based on the passed in settings
        :return transforms [dict] : dictionary of transforms (for training and test sets)
        """
        transforms = {'train': tt.ToTensor(), 'test': tt.ToTensor()}  # default
        trainTransforms = []
        testTransforms = []
        # construct training transforms
        if "hflip" in self.dataAugmentationSettings["train"]:
            trainTransforms.append(tt.RandomHorizontalFlip())
        if "vflip" in self.dataAugmentationSettings["train"]:
            trainTransforms.append(tt.RandomVerticalFlip())
        trainTransforms.append(tt.ToTensor())
        transforms["train"] = tt.Compose(trainTransforms)
        # construct testing transforms
        if "hflip" in self.dataAugmentationSettings["test"]:
            testTransforms.append(tt.RandomHorizontalFlip())
        if "vflip" in self.dataAugmentationSettings["test"]:
            testTransforms.append(tt.RandomVerticalFlip())
        testTransforms.append(tt.ToTensor())
        transforms["test"] = tt.Compose(testTransforms)
        print(type(transforms))
        return transforms
