from matplotlib import pyplot as plt


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

    def rawExamplesOneDataLoader(self, dims, deviceDataLoader):
        """
        create a collage of data the size of passed in dims for both training and testing data
        :arg dims [[int, int]] : number of rows and cols of images to prepare
        :return figure [plt.figure] : collage of desired dimensions for a set from given dataloader
        """
        idxToClass = dict(map(reversed, deviceDataLoader.dataLoader.dataset.class_to_idx.items()))
        numRows, numCols = dims
        numTotal = numRows * numCols
        cacheImages = []
        cacheLabels = []
        for images, labels in deviceDataLoader:
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
            ax.tick_params(axis="x", labelsize=3)
            ax.tick_params(axis="y", labelsize=3)
            ax.set_title(idxToClass.get(int(label)), fontsize=5)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        return fig

    def rawExamples(self, dims, fileSuffix):
        """
        create a collage of data the size of passed in dims for both training and testing data
        :arg dims [[int, int]] : number of rows and cols of images to prepare
        :return figure [[plt.figure, plt.figure]] : collage of desired dimensions for training and testing set
        """
        trainFig = self.rawExamplesOneDataLoader(dims, self.datasetLoader.trainDeviceDataLoader)
        trainFig.savefig(f"edaResults/rawExamplesTraining{fileSuffix}.png")
        testFig = self.rawExamplesOneDataLoader(dims, self.datasetLoader.testDeviceDataLoader)
        testFig.savefig(f"edaResults/rawExamplesTesting{fileSuffix}.png")
        return True

    def getNumDatapoints(self):
        """
        :return numTrain, numTest [int, int] : number of datapoints in training and test set
        """
        numTrain = len(self.datasetLoader.trainDeviceDataLoader)
        numTest = len(self.datasetLoader.testDeviceDataLoader)
        return [numTrain, numTest]

    def getClasses(self):
        pass

    def getMean(self):
        pass

    def getStd(self):
        pass
