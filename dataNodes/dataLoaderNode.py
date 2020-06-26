from executionNodes.loggerNode import logs
from executionNodes.deviceManager import deviceManager as DM
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


class DatasetLoader():
    def __init__(self, dataDir, batchSize=4, numWorkers=4,
                 dataAugmentationObj={'train': transforms.ToTensor(), 'test': transforms.ToTensor()}):
        """
        constructor
        manages the reading of raw data and preparing it into a PyTorch train & test datasets
        :attr trainDataset [torch.utils.data.Dataset] : the dataset of training images
        :attr testDataset [torch.utils.data.Dataset] : the dataset of testing images
        """
        trainDir = dataDir + '/train'
        testDir = dataDir + '/test'
        try:
            trainDataset = datasets.ImageFolder(root=trainDir, transform=dataAugmentationObj['train'])
            testDataset = datasets.ImageFolder(root=testDir, transform=dataAugmentationObj['test'])
            trainDataLoader = DataLoader(trainDataset,
                                         batch_size=batchSize,
                                         shuffle=True,
                                         num_workers=numWorkers)
            testDataLoader = DataLoader(testDataset,
                                        batch_size=batchSize,
                                        shuffle=True,
                                        num_workers=numWorkers)
        except Exception as e:
            logs.dataLoader.error("dataset creation unsuccessful: %s", e)
        self.trainDeviceDataLoader = DeviceDataLoader(trainDataLoader)
        self.testDeviceDataLoader = DeviceDataLoader(testDataLoader)


class DeviceDataLoader():
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def __iter__(self):
        for batch in self.dataloader:
            yield DM.moveToDevice(batch)

    def __len__(self):
        return len(self.dataloader)
