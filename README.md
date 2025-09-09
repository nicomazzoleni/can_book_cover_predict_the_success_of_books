# Book Cover Analysis and Statistical Insights

## Overview

This repository contains two Jupyter Notebooks focused on analyzing book cover images and their visual features, with an emphasis on insider vs outsider dynamics in the publishing industry.

1. **`stat_analysis_brief.ipynb`**: Performs statistical analysis on extracted features, comparing "insiders" and "outsiders" (based on a `special_feature` flag) to identify patterns in romance book covers.
2. **`book_cover_analysis_and_prediction.ipynb`**: Extracts visual features (e.g., brightness, saturation, object counts) from book cover images using computer vision techniques and analyses data to discern the effect of insider vs outsider status.

The notebooks aim to understand visual characteristics of book covers and their relationship to genres or specific attributes, enabling potential applications like genre classification or marketing insights.

## Objective

The primary objectives are:
- Extract and analyze visual features from book cover images (e.g., brightness, contrast, detected objects).
- Compare feature distributions between groups (insiders vs. outsiders) in the romance genre.
- Prepare data for predictive tasks, such as genre classification or identifying special cover characteristics.
- Provide statistical insights to highlight differences in cover designs.

## Data Description

The analysis uses a dataset of book covers, likely sourced from Goodreads or similar, with the following key components:
- **Dataset**: A DataFrame (e.g., `romance_df`) containing book metadata and extracted features.
- **Key Variables**:
  - `image_url`: URL to the book cover image.
  - `special_feature`: Binary flag (1 for insiders, 0 for outsiders, specific meaning not detailed).
  - Numerical features: `brightness`, `saturation`, `color_count`, `colorfulness`, `contrast`, `edge_density`, `num_objects`.
  - Binary object detection flags: `is_person`, `is_bird`, `is_clock`, `is_dog`, `is_cup`, `is_tie`, `is_cat`, `is_tv`.
- Images are processed on-the-fly using URLs, with features extracted via computer vision.

## Notebook Structure

### `book_cover_analysis_and_prediction.ipynb`
1. **Data Loading**:
   - Installs Ultralytics (YOLO) for object detection.
   - Loads book dataset (e.g., romance genre DataFrame).
2. **Image Processing and Feature Extraction**:
   - Downloads cover images from URLs.
   - Computes visual metrics (brightness, saturation, etc.) using OpenCV.
   - Detects objects using YOLO pre-trained models.
3. **Analysis and Examples**:
   - Filters and displays features for specific books (e.g., a romance book with `image_url`).
   - Shows DataFrames with extracted metrics (e.g., brightness ~0.242, `is_person=1`).

### `stat_analysis_brief.ipynb`
1. **Data Loading**:
   - Loads romance dataset with pre-extracted features.
2. **Group Splitting**:
   - Splits data into insiders (`special_feature == 1`) and outsiders (`special_feature == 0`).
3. **Statistical Analysis**:
   - Computes means and std devs for `num_objects`.
   - Calculates proportions for binary features (e.g., `is_person`).
   - Visualizes proportions with bar plots.
4. **Results Display**:
   - Shows DataFrames with numerical stats and binary proportions.

## Dependencies

To run the notebooks, ensure the following Python libraries are installed:

- `ultralytics` (for YOLO object detection)
- `numpy`, `matplotlib`, `opencv-python`, `pillow`, `pyyaml`, `requests`, `scipy`, `torch`, `torchvision`, `tqdm`, `psutil`, `pandas`, `seaborn`

Install them using:

```bash
pip install ultralytics numpy matplotlib opencv-python pillow pyyaml requests scipy torch torchvision tqdm psutil pandas seaborn
