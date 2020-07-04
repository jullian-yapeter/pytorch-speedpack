from matplotlib import pyplot as plt
import numpy as np


class EdaManager():
    def __init__(self, datasetLoader):
        """
        constructor
        tools for the automatic exploratory data analysis of the passed in dataset
        :arg datasetLoader [DatasetLoader] : the dataset loader of the data to be explored
        :attr datasetLoader [DatasetLoader] : the dataset loader of the data to be explored
        """
        self.datasetLoader = datasetLoader

    def show(self, figure):
        pass

    def save(self, figure):
        pass

    def rawExamples(self, dims):
        """
        create a collage of data the size of passed in dims for both training and testing data
        :arg dims [[int, int]] : number of rows and cols of images to prepare
        :return figure [[plt.figure, plt.figure]] : collage of desired dimensions for training and testing set
        """
        numRows, numCols = dims
        numTotal = numRows * numCols
        cacheImages = []
        cacheLabels = []
        for images, labels in self.datasetLoader.trainDeviceDataLoader:
            cacheImages.extend(images)
            cacheLabels.extend(labels)
            if len(cacheImages) >= numTotal:
                break
        fig = plt.figure()
        for n, (image, label) in enumerate(zip(cacheImages, cacheLabels)):
            if n >= numTotal:
                break
            image = image.permute(1, 2, 0)
            ax = fig.add_subplot(numCols, numRows, n + 1)
            if image.ndim == 2:
                plt.gray()
            ax.imshow(image)
            ax.set_title(label)
        fig.set_size_inches(np.array(fig.get_size_inches()) * numTotal)
        fig.savefig("edaResults/rawExamplesTraining.png")
        return True

    def augmentedExamples(self):
        pass

    def getNumDatapoints(self):
        pass

    def getClasses(self):
        pass

    def getMean(self):
        pass

    def getStd(self):
        pass
