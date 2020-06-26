import logging
import logging.handlers


class Logs():
    def __init__(self):
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
        self.webInterface = self.setup_logger(name='webInterface', log_file='logs/webInterface.log',
                                              formatter=formatter)

    def setup_logger(self, name, log_file, formatter, level=logging.INFO):
        handler = logging.handlers.RotatingFileHandler(log_file, mode='a', maxBytes=5*1024*1024,
                                                       backupCount=2, encoding=None, delay=0)
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)

        return logger


logs = Logs()
