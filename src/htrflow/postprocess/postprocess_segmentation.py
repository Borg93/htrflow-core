import cv2
import numpy as np
import torch

from htrflow.structures.result import Result
from htrflow.utils.helper import timing_decorator


class PostProcessSegmentation:
    def __init__(self):
        pass

    @staticmethod
    @timing_decorator
    def crop_imgs_from_result(result: Result, img):
        cropped_imgs = []

        masks = result.segmentation.masks.cpu().numpy().astype(np.uint8)
        # try with the bounding box function instead and compare with optim
        for j, mask in enumerate(masks):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            largest_contour = max(contours, key=cv2.contourArea)

            # epsilon = 0.003 * cv2.arcLength(largest_contour, True)
            # approx_poly = cv2.approxPolyDP(largest_contour, epsilon, True)
            # approx_poly = np.squeeze(approx_poly)
            # approx_poly = approx_poly.tolist()
            # polygons.append(approx_poly)

            x, y, w, h = cv2.boundingRect(largest_contour)

            # Crop masked region and put on white background
            masked_region = img[y : y + h, x : x + w]
            white_background = np.ones_like(masked_region)
            white_background.fill(255)

            mask_roi = mask[y : y + h, x : x + w]

            masked_region_on_white = cv2.bitwise_and(white_background, masked_region, mask=mask_roi)

            cv2.bitwise_not(white_background, white_background, mask=mask_roi)
            res = white_background + masked_region_on_white

            cropped_imgs.append(res)

        return cropped_imgs

    def get_bounding_box(mask):
        rows = torch.any(mask, dim=1)
        cols = torch.any(mask, dim=0)
        ymin, ymax = torch.where(rows)[0][[0, -1]]
        xmin, xmax = torch.where(cols)[0][[0, -1]]

        return xmin, ymin, xmax, ymax

    def get_bounding_box_np(mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]

        return xmin, ymin, xmax, ymax

    @staticmethod
    @timing_decorator
    def crop_imgs_from_result_optim(result: Result, img):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Convert img to a PyTorch tensor and move to GPU if available
        img = torch.from_numpy(img).to(device)

        cropped_imgs = []
        masks = result.segmentation.masks.to(device)

        for mask in masks:
            # Get bounding box
            xmin, ymin, xmax, ymax = PostProcessSegmentation.get_bounding_box(mask)

            # Crop masked region and put on white background
            masked_region = img[ymin : ymax + 1, xmin : xmax + 1]
            white_background = torch.ones_like(masked_region) * 255

            # Apply mask to the image
            masked_region_on_white = torch.where(
                mask[ymin : ymax + 1, xmin : xmax + 1][..., None], masked_region, white_background
            )
            masked_region_on_white_np = masked_region_on_white.cpu().numpy()

            cropped_imgs.append(masked_region_on_white_np)

        return cropped_imgs
