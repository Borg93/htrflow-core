import mmcv

from htr_svea.inferencer.base_inferencer import BaseInferencer
from mmdet.structures import DetDataSample


class MMOCRInferencer(BaseInferencer):
    def __init__(self, region_model):
        self.region_model = region_model
        self.raw_pred_result = DetDataSample()

    def preprocess():
        pass

    def predict(self, input_image):
        image = mmcv.imread(input_image)

        
        return image

    def postprocess():
        pass