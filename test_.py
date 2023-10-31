import os
from glob import glob

import mmcv

from htr_svea.inferencer.mmdet_inferencer import MMDetInferencer
from htr_svea.models.openmmlab_models import OpenmmlabModel
from htr_svea.postprocess.postprocess_segmentation import PostProcessSegmentation
from htr_svea.utils.helper import timing_decorator


def post_process_seg(result, imgs, lines = False, regions = False):

    imgs_cropped = list()
    
    for res, img in zip(result, imgs):
        res.segmentation.remove_overlapping_masks()
        res.segmentation.align_masks_with_image(img)
        
        indices = False
        if regions:
            res = PostProcessSegmentation.order_regions_marginalia(res)
        elif lines:
            res = PostProcessSegmentation.order_lines(res)
        
        imgs_cropped.append(PostProcessSegmentation.crop_imgs_from_result_optim(res, img))

    return result, imgs_cropped

def post_process_seg(result, imgs, lines = False, regions = False):

    imgs_cropped = list()
    
    for res, img in zip(result, imgs):
        res.segmentation.remove_overlapping_masks()
        res.segmentation.align_masks_with_image(img)
        
        indices = False
        if regions:
            res = PostProcessSegmentation.order_regions_marginalia(res)
        elif lines:
            res = PostProcessSegmentation.order_lines(res)
        
        imgs_cropped.append(PostProcessSegmentation.crop_imgs_from_result_optim(res, img))

    return result, imgs_cropped

@timing_decorator
def predict_batch(inferencer_regions, inferencer_lines, imgs_numpy):
    result_regions = inferencer_regions.predict(imgs_numpy, batch_size=8)

    imgs_region_numpy = list()

    result_regions, imgs_region_numpy = post_process_seg(result_regions, imgs_numpy, regions=True)
    flat_imgs_region_numpy = [item for sublist in imgs_region_numpy for item in sublist]
    result_lines = inferencer_lines.predict(flat_imgs_region_numpy, batch_size=8)
    result_lines, imgs_lines_numpy = post_process_seg(result_lines, flat_imgs_region_numpy, lines=True)

    return True


if __name__ == "__main__":
    region_model = OpenmmlabModel.from_pretrained("Riksarkivet/rtmdet_regions", cache_dir="./config")

    print(region_model)

    lines_model = OpenmmlabModel.from_pretrained("Riksarkivet/rtmdet_lines", cache_dir="./config")

    # try with region_inferencer instead (type=region till mmdetinferencer)

    inferencer_regions = MMDetInferencer(region_model=region_model)
    inferencer_lines = MMDetInferencer(region_model=lines_model)

    imgs = glob(
        os.path.join(
            "/media/erik/Elements/Riksarkivet/data/datasets/htr/Trolldomskommissionen3/Kommissorialrätt_i_Stockholm_ang_trolldomsväsendet,_nr_4_(1676)",
            "**",
            "bin_image",
            "*",
        ),
        recursive=True,
    )
    imgs_numpy = []

    for img in imgs[0:8]:
        imgs_numpy.append(mmcv.imread(img))
    
    predict_batch(inferencer_regions, inferencer_lines, imgs_numpy)

    
    
    #print(result[-1].img_shape)
    #print(result['predictions'][0].pred_instances.metadata_fields)
    #print(result['predictions'][0]._metainfo_fields)
    #print(result.keys())
    #print(result)    

    from PIL import Image

    # load image from the IAM database
    # image = Image.open("./image_0.png").convert("RGB")

    # Use a pipeline as a high-level helper
    # from transformers import pipeline

    # pipe = pipeline("image-to-text", model="microsoft/trocr-large-handwritten")
    # print(pipe(image, batch_size=8))
