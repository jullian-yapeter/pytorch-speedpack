import torch


class DeviceManager():
    def __init__(self):
        """
        constructor
        manages the selection and loading of computing device
        """
        self.device = self.getDefaultDevice()

    def getDefaultDevice(self):
        """
        select the best device available
        :return [torch.device] : selected device
        """
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def moveToDevice(self, data):
        """
        select the best device available
        :return [torch.Tensor] : moved data
        """
        if isinstance(data, (list, tuple)):
            return [self.moveToDevice(d) for d in data]
        return data.to(self.device, non_blocking=True)
