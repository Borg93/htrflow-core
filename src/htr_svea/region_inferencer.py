import mmcv

from src.htr_svea.model_collector import HtrModels


class RegInferencer:
    def __init__(self, region_model: HtrModels):
        self.region_model = region_model

    def predict(self, input_image):
        image = mmcv.imread(input_image)
        bin_image = self.preprocess_img(image)

        print(bin_image)
