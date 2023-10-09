from htr_svea.model_collector import OpenmmlabModel

if __name__ == "__main__":
    region_model = OpenmmlabModel.from_pretrained("Riksarkivet/rmtdet_region", cache_dir="./config")

    print(region_model)


https://github.com/open-mmlab/mmocr/blob/main/mmocr/apis/inferencers/base_mmocr_inferencer.py#L26
https://mmocr.readthedocs.io/en/stable/_modules/mmocr/apis/inferencers/mmocr_inferencer.html
