import cv2
import numpy as np
import torch
import kornia
import pandas as pd

from htr_svea.structures.result import Result
from htr_svea.utils.helper import timing_decorator

class PostProcessSegmentation():

    def __init__(self):
        pass

    @staticmethod
    @timing_decorator
    def crop_imgs_from_result(result: Result, img):
        cropped_imgs = list()
        polygons = list()

        masks = result.segmentation.masks.cpu().numpy().astype(np.uint8)
        #try with the bounding box function instead and compare with optim
        for j, mask in enumerate(masks):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            largest_contour = max(contours, key=cv2.contourArea)

            #epsilon = 0.003 * cv2.arcLength(largest_contour, True)
            #approx_poly = cv2.approxPolyDP(largest_contour, epsilon, True)
            #approx_poly = np.squeeze(approx_poly)
            #approx_poly = approx_poly.tolist()
            #polygons.append(approx_poly)

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

        cropped_imgs = list()
        masks = result.segmentation.masks.to(device)

        for mask in masks:
            # Get bounding box
            xmin, ymin, xmax, ymax = PostProcessSegmentation.get_bounding_box(mask)

            # Crop masked region and put on white background
            masked_region = img[ymin:ymax+1, xmin:xmax+1]
            white_background = torch.ones_like(masked_region) * 255

            # Apply mask to the image
            masked_region_on_white = torch.where(mask[ymin:ymax+1, xmin:xmax+1][..., None], masked_region, white_background)
            masked_region_on_white_np = masked_region_on_white.cpu().numpy()

            cropped_imgs.append(masked_region_on_white_np)

        return cropped_imgs
    
    def _order_seg_res_with_ind(result, indices):
        result.segmentation.labels = torch.stack([result.segmentation.labels[i] for i in indices])
        result.segmentation.scores = torch.stack([result.segmentation.scores[i] for i in indices])
        result.segmentation.bboxes = torch.stack([result.segmentation.bboxes[i] for i in indices])
        result.segmentation.masks = torch.stack([result.segmentation.masks[i] for i in indices])

        return result
    
    @staticmethod
    @timing_decorator
    def _calculate_threshold_distance(bounding_boxes, line_spacing_factor=0.5):
        # Calculate the average height of the text lines
        total_height = sum(box[3] - box[1] for box in bounding_boxes)
        average_height = total_height / len(bounding_boxes)

        # Calculate the threshold distance, Set a factor for the threshold distance (adjust as needed)
        threshold_distance = average_height * line_spacing_factor

        # Return the threshold distance
        return threshold_distance
    
    @staticmethod
    @timing_decorator
    def order_lines(result, line_spacing_factor=0.5):
        bounding_boxes = result.segmentation.bboxes.tolist()
        center_points = [(box[1] + box[3]) / 2 for box in bounding_boxes]
        horizontal_positions = [(box[0] + box[2]) / 2 for box in bounding_boxes]

        # Calculate the threshold distance
        threshold_distance = PostProcessSegmentation._calculate_threshold_distance(bounding_boxes, line_spacing_factor)

        # Sort the indices based on vertical center points and horizontal positions
        indices = list(range(len(bounding_boxes)))
        indices.sort(
            key=lambda i: (
                center_points[i] // threshold_distance,
                horizontal_positions[i],
            )
        )

        # Order text lines
        result = PostProcessSegmentation._order_seg_res_with_ind(result, indices)

        return result
    
    @staticmethod
    @timing_decorator
    def order_regions_marginalia(result, margin_ratio=0.2, histogram_bins=50, histogram_dip_ratio=0.5):
        bounding_boxes = result.segmentation.bboxes.tolist()
        img_width = result.img_shape[0]  #is it really [0]

        regions = [[i, x[0], x[1], x[0] + x[2], x[1] + x[3]] for i, x in enumerate(bounding_boxes)]

        # Create a pandas DataFrame from the regions
        df = pd.DataFrame(regions, columns=["region_id", "x_min", "y_min", "x_max", "y_max"])

        # Calculate the centroids of the bounding boxes
        df["centroid_x"] = (df["x_min"] + df["x_max"]) / 2
        df["centroid_y"] = (df["y_min"] + df["y_max"]) / 2

        # Calculate a histogram of the x-coordinates of the centroids
        histogram, bin_edges = np.histogram(df["centroid_x"], bins=histogram_bins)

        # Determine if there's a significant dip in the histogram, which would suggest a two-page layout
        is_two_pages = np.min(histogram) < np.max(histogram) * histogram_dip_ratio

        if is_two_pages:
            # Determine which page each region is on
            page_width = int(img_width / 2)
            df["page"] = (df["centroid_x"] > page_width).astype(int)

            # Determine if the region is in the margin
            margin_width = page_width * margin_ratio
            df["is_margin"] = ((df["page"] == 0) & (df["centroid_x"] < margin_width)) | (
                (df["page"] == 1) & (df["centroid_x"] > img_width - margin_width)
            )
        else:
            df["page"] = 0
            df["is_margin"] = (df["centroid_x"] < img_width * margin_ratio) | (
                df["centroid_x"] > img_width - page_width * margin_ratio
            )

        # Define a custom sorting function
        sort_regions = lambda row: (
            row["page"],
            row["is_margin"],
            row["centroid_y"],
            row["centroid_x"],
        )

        # Sort the DataFrame using the custom function
        df["sort_key"] = df.apply(sort_regions, axis=1)
        df = df.sort_values("sort_key")

        # Return the ordered regions
        indices = df["region_id"].tolist()

        result = PostProcessSegmentation._order_seg_res_with_ind(result, indices)
        return result