import logging

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from mmdet.apis import DetInferencer
from mmengine.config import Config
from mmocr.apis import TextRecInferencer

LIB_TEMP_NAME = "HTRansform"
REPO_TYPE = "model"
MODEL_FILE = "model.pth"
CONFIG_FILE = "config.py"


class OpenmmlabModel:
    def __init__(self, config_files, model_files):
        self.config_files = config_files
        self.model_files = model_files
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @classmethod
    def from_pretrained(cls, model_id: str, cache_dir: str = None, device: str = None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_files, config_files = cls.download_files(model_id, cache_dir)
        if model_files and config_files:
            cfg = Config.fromfile(config_files)
            model = ModelFactory.create_model(cfg, config_files, model_files, device)
            return model
        return None

    def from_local(self):
        cfg = Config.fromfile(self.config_files)
        model = ModelFactory.create_model(cfg, self.config_files, self.model_files, self.device)
        return model

    @staticmethod
    def download_files(repo_id, cache_dir):
        try:
            model_files = hf_hub_download(
                repo_id=repo_id,
                repo_type=REPO_TYPE,
                filename=MODEL_FILE,
                library_name=LIB_TEMP_NAME,
                cache_dir=cache_dir,
            )
            config_files = hf_hub_download(
                repo_id=repo_id,
                repo_type=REPO_TYPE,
                filename=CONFIG_FILE,
                library_name=LIB_TEMP_NAME,
                cache_dir=cache_dir,
            )
            return model_files, config_files
        except RepositoryNotFoundError as e:
            logging.error(f"Could not download files for {repo_id}: {str(e)}")
            return None, None


class ModelFactory:
    @staticmethod
    def create_model(cfg, config_files, model_files, device):
        model_creators = {
            "mmdet": DetInferencer,
            "mmocr": TextRecInferencer,
        }
        model_scope = cfg.default_scope
        if model_scope in model_creators:
            return model_creators[model_scope](config_files, model_files, device=device)
        logging.error(f"Unknown model scope: {model_scope}")
        return None


# TODO  read config and update the cfg.dict path (i.e dump a new file and init the textrecinferencer based on the new file.)
