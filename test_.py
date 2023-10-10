from htr_svea.openmmlab_models import OpenmmlabModel


if __name__ == "__main__":
    region_model = OpenmmlabModel.from_pretrained("Riksarkivet/rtmdet_regions", cache_dir="./config")

    print(region_model)
