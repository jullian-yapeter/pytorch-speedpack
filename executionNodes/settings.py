from executionNodes.loggerNode import logs
import json


class SettingsManager():
    def __init__(self):
        """
        constructor
        :attr settings [dict] : dictionary of user defined settings for Speedpack
        """
        try:
            with open('settings.json') as settingsFile:
                self.settings = json.load(settingsFile)
            logs.settings.info("Settings loaded successfully")
        except Exception as e:
            logs.settings.error("Error while opening settings file, %s", e)

    def readField(self, field):
        """
        :return [String] : data of a certain field within the settings
        """
        if field in self.settings.keys():
            return self.settings.get(field)
        else:
            logs.settings.error("No field in settings called %s", str(field))
