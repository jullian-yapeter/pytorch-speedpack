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
        :arg  dataDir [String] : directory of data
        :arg batchSize [int] : number of datapoints for one batch of training
        :arg numWorkers [int] : number of workers to perform the data loading
        :arg dataAugmentationObj [dict] : instructions for data augmentation to be performed during training/testing
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
    def __init__(self, dataLoader):
        """
        constructor
        wraps the PyTorch dataloader in class that handles the moving of data onto the optimal device
        :attr dataloader [torch.utils.data.DataLoader] : the PyTorch Dataloader of a dataset to be wrapped
        """
        self.dataLoader = dataLoader

    def __iter__(self):
        """
        Generates a batch of data from the given Dataloader that has been moved to the optimal device
        :yield [torch.Tensor] : moved batch of data
        """
        for batch in self.dataLoader:
            yield DM.moveToDevice(batch)

    def __len__(self):
        return len(self.dataLoader)
