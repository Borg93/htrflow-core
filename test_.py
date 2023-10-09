from htr_svea.model_collector import OpenmmlabModel

if __name__ == "__main__":
    region_model = OpenmmlabModel.from_pretrained("Riksarkivet/rmtdet_region", cache_dir="./config")

    print(region_model)
