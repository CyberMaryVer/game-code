import os
import yaml


# TODO add _check_config (if_file_exist, if_all_conditions_have_same_length)
class GameConfigLoader:
    def __init__(self, config):
        self._set_config(config)

    def _set_config(self, config):
        cfg_dict = self.read_cfg_file(config)
        self.WINDOW_NAME = cfg_dict["WINDOW_NAME"]
        self.DESCRIPTION = cfg_dict["DESCRIPTION"]
        self.KEYPOINTS_FOR_GAME = cfg_dict["KEYPOINTS_FOR_GAME"]
        self.GAMES_NUMBER = cfg_dict["GAMES_NUMBER"]
        self.SCALE = cfg_dict["SCALE"]
        self.IMAGE_SIZE = cfg_dict["IMAGE_SIZE"]
        self.SAVE_OUTPUT = cfg_dict["SAVE_OUTPUT"]
        self.SAVE_PATH = cfg_dict["SAVE_PATH"]
        self.IMAGE_PATH = cfg_dict["IMAGE_PATH"]
        self.IMAGE_TAGS = cfg_dict["IMAGE_TAGS"]
        self.IMAGE_CAT = cfg_dict["IMAGE_CAT"]
        self.META_WIN = cfg_dict["META_WIN"]
        self.META_FAIL = cfg_dict["META_FAIL"]
        self.SPEED = cfg_dict["SPEED"]
        self.ACCELERATION = cfg_dict["ACCELERATION"]
        self.PAUSE_MAX = cfg_dict["PAUSE_MAX"]

    @staticmethod
    def _get_path(path):
        if __name__ == "__main__":
            folder = ""
        else:
            folder = "game_configs"
        return os.path.join(folder, path)

    @classmethod
    def rzd(cls):
        config = "rzd.yaml"
        path = cls._get_path(config)
        return cls(path)

    @classmethod
    def princess(cls):
        config = "scp.yaml"
        path = cls._get_path(config)
        return cls(path)

    @classmethod
    def covid(cls):
        config = "covid.yaml"
        path = cls._get_path(config)
        return cls(path)

    @staticmethod
    def read_cfg_file(config):
        with open(config) as info:
            info_dict = yaml.load(info, Loader=yaml.FullLoader)
        return info_dict


if __name__ == "__main__":
    w = GameConfigLoader.rzd()
    print(w.DESCRIPTION, type(w.DESCRIPTION))
