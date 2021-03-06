import logging
import logging.handlers


class Logs():
    def __init__(self):
        """
        constructor
        creates the loggers for all the different modules in one encapsulated object
        :attr dataLoader        [logging.Logger] : logger #1  for dataLoaderNode
        :attr eda               [logging.Logger] : logger #2  for edaNode
        :attr dataAugmentation  [logging.Logger] : logger #3  for dataAugmentationNode
        :attr modelFeature      [logging.Logger] : logger #4  for modelFeatureNode
        :attr modelHead         [logging.Logger] : logger #5  for modelHeadNode
        :attr model             [logging.Logger] : logger #6  for modelNode
        :attr loss              [logging.Logger] : logger #7  for lossNode
        :attr optimizer         [logging.Logger] : logger #8  for optimizerNode
        :attr resultsFolder     [logging.Logger] : logger #9  for resultsFolderNode
        :attr autoDocumentation [logging.Logger] : logger #10 for autoDocumentationNode
        :attr interface         [logging.Logger] : logger #11 for interfaceNode
        :attr debugger          [logging.Logger] : logger #12 for debuggingNodes
        """
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        self.dataLoader = self.setup_logger(name='dataLoader', log_file='logs/dataLoader.log', formatter=formatter)
        self.eda = self.setup_logger(name='eda', log_file='logs/eda.log', formatter=formatter)
        self.dataAugmentation = self.setup_logger(name='dataAugmentation', log_file='logs/dataAugmentation.log',
                                                  formatter=formatter)
        self.modelFeature = self.setup_logger(name='modelFeature', log_file='logs/modelFeature.log',
                                              formatter=formatter)
        self.modelHead = self.setup_logger(name='modelHead', log_file='logs/modelHead.log', formatter=formatter)
        self.model = self.setup_logger(name='model', log_file='logs/model.log', formatter=formatter)
        self.loss = self.setup_logger(name='loss', log_file='logs/loss.log', formatter=formatter)
        self.optimizer = self.setup_logger(name='optimizer', log_file='logs/optimizer.log', formatter=formatter)
        self.resultsFolder = self.setup_logger(name='resultsFolder', log_file='logs/resultsFolder.log',
                                               formatter=formatter)
        self.autoDocumentation = self.setup_logger(name='autoDocumentation', log_file='logs/autoDocumentation.log',
                                                   formatter=formatter)
        self.interface = self.setup_logger(name='interface', log_file='logs/interface.log', formatter=formatter)
        self.debugging = self.setup_logger(name='debugging', log_file='logs/debugging.log', formatter=formatter)
        self.settings = self.setup_logger(name='settings', log_file='logs/settings.log', formatter=formatter)

    def setup_logger(self, name, log_file, formatter, level=logging.INFO):
        """
        helper function to set up the behaviour of the logger
        :return [logging.Logger] : rolling logger with desired name and behaviour
        """
        handler = logging.handlers.RotatingFileHandler(log_file, mode='a', maxBytes=5*1024*1024,
                                                       backupCount=2, encoding=None, delay=0)
        handler.setFormatter(formatter)
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        return logger


logs = Logs()
