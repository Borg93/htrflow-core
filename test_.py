# from htr_svea.models.openmmlab_models import OpenmmlabModel

if __name__ == "__main__":
    # region_model = OpenmmlabModel.from_pretrained("Riksarkivet/rtmdet_regions", cache_dir="./config")

    # print(region_model)

    from PIL import Image

    # load image from the IAM database
    image = Image.open("./image_0.png").convert("RGB")

    # Use a pipeline as a high-level helper
    from transformers import pipeline

    pipe = pipeline("image-to-text", model="microsoft/trocr-large-handwritten")
    print(pipe(image, batch_size=8))
