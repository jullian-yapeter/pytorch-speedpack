import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as tt

from executionNodes.loggerNode import logs
from executionNodes.deviceManager import deviceManager as DM


def classificationCollate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


class DatasetLoader():
    def __init__(self, dataDir, batchSize=4, numWorkers=4, collateFn=None,
                 dataAugmentationObj={'train': tt.ToTensor(), 'test': tt.ToTensor()}):
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
                                         collate_fn=collateFn,
                                         shuffle=True,
                                         num_workers=numWorkers)
            testDataLoader = DataLoader(testDataset,
                                        batch_size=1,
                                        shuffle=True,
                                        num_workers=numWorkers)
        except Exception as e:
            logs.dataLoader.error("dataset loader creation unsuccessful: %s", e)
        self.trainDeviceDataLoader = DeviceDataLoader(trainDataLoader)
        self.testDeviceDataLoader = DeviceDataLoader(testDataLoader)
        logs.dataLoader.info("dataset loader creation successful")


class DeviceDataLoader():
    def __init__(self, dataLoader):
        """
        constructor
        wraps the PyTorch dataloader in a class that handles the moving of data onto the optimal device
        :arg dataLoader [torch.utils.data.DataLoader] : the PyTorch Dataloader of a dataset to be wrapped
        :attr dataLoader [torch.utils.data.DataLoader] : the PyTorch Dataloader of a dataset to be wrapped
        """
        self.dataLoader = dataLoader

    def __iter__(self):
        """
        generates a batch of data from the given Dataloader that has been moved to the optimal device
        :yield [torch.Tensor] : moved batch of data
        """
        for batch in self.dataLoader:
            yield DM.moveToDevice(batch)

    def __len__(self):
        return len(self.dataLoader)
