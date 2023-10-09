import mmcv
from mmdet.apis import DetInferencer


class RegInferencer:
    def __init__(self, region_model: DetInferencer):
        self.region_model = region_model

    def preprocess():
        pass

    def predict(self, input_image):
        image = mmcv.imread(input_image)
        bin_image = self.preprocess_img(image)

        print(bin_image)

    def postprocess():
        pass
