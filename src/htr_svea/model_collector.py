import logging
from enum import Enum

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from mmdet.apis import DetInferencer
from mmengine.config import Config
from mmocr.apis import TextRecInferencer


class OpenmmlabsFile(Enum):
    MODEL_FILE = "model.pth"
    CONFIG_FILE = "config.py"
    DICT_FILE = "dictionary.txt"


class OpenmmlabsFramework(Enum):
    MMDET = "mmdet"
    MMOCR = "mmocr"


class OpenmmlabModel:
    REPO_TYPE = "model"

    def __init__(self, config_files, model_files):
        self.config_files = config_files
        self.model_files = model_files
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @classmethod
    def from_pretrained(cls, model_id: str, cache_dir: str = None, device: str = None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_file, config_file = cls.download_config_and_model_file(model_id, cache_dir)

        if model_file and config_file:
            model_scope = cls._checking_model_scope(model_id, cache_dir, config_file)

            model = OpenModelFactory.create_openmmlab_model(model_scope, config_file, model_file, device)
            return model
        return None

    @classmethod
    def _checking_model_scope(cls, model_id, cache_dir, config_file):
        cfg = Config.fromfile(config_file)
        model_scope = cfg.default_scope

        if model_scope == OpenmmlabsFramework.MMOCR.value:
            download_dict_file = cls.download_config_and_model_file(model_id, cache_dir)
            cfg.dictionary["dict_file"] = download_dict_file

            cfg.dump(config_file)
        return model_scope

    def from_local(self):
        cfg = Config.fromfile(self.config_files)
        model = OpenModelFactory.create_openmmlab_model(cfg, self.config_files, self.model_files, self.device)
        return model

    @staticmethod
    def download_config_and_model_file(repo_id, cache_dir):
        try:
            model_file = hf_hub_download(
                repo_id=repo_id,
                repo_type=OpenmmlabModel.REPO_TYPE,
                filename=OpenmmlabsFile.MODEL_FILE.value,
                library_name=__package__,
                cache_dir=cache_dir,
            )
            config_file = hf_hub_download(
                repo_id=repo_id,
                repo_type=OpenmmlabModel.REPO_TYPE,
                filename=OpenmmlabsFile.CONFIG_FILE.value,
                library_name=__package__,
                cache_dir=cache_dir,
            )
            return model_file, config_file

        except RepositoryNotFoundError as e:
            logging.error(f"Could not download files for {repo_id}: {str(e)}")
            return None, None

    @staticmethod
    def download_dict_file(repo_id, cache_dir):
        try:
            dictionary_file = hf_hub_download(
                repo_id=repo_id,
                repo_type=OpenmmlabModel.REPO_TYPE,
                filename=OpenmmlabsFile.DICT_FILE.value,
                library_name=__package__,
                cache_dir=cache_dir,
            )

            return dictionary_file

        except RepositoryNotFoundError as e:
            logging.error(f"Could not download files for {repo_id}: {str(e)}")
            return None


class OpenModelFactory:
    @staticmethod
    def create_openmmlab_model(model_scope, config_files, model_files, device):
        model_creators = {
            OpenmmlabsFramework.MMDET.value: DetInferencer,
            OpenmmlabsFramework.MMOCR.value: TextRecInferencer,
        }

        if model_scope in model_creators:
            return model_creators[model_scope](config_files, model_files, device=device)
        logging.error(f"Unknown model scope: {model_scope}")
        return None
