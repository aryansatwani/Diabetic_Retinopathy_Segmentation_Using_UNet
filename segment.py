import streamlit as st
import pandas as pd
from PIL import Image
import os

# Function to load segmentation results from Excel file
def load_segmentation_results(file_path):
    # Read Excel file and append full file paths to image paths
    segmentation_results = pd.read_excel(file_path)
    segmentation_results['Input'] = segmentation_results['Input'].apply(lambda x: os.path.join("E:/diabetic_retinopathy_input/", x))
    segmentation_results['Output'] = segmentation_results['Output'].apply(lambda x: os.path.join("E:/diabetic_retinopathy_test/", x))
    return segmentation_results

# Function to display segmentation results and metrics
def display_results(image, segmentation, metrics):
    st.image(image, caption='Original Image', use_column_width=True)
    st.image(segmentation, caption='Segmented Image', use_column_width=True)
    st.subheader("Performance Metrics")
    st.write(metrics)


# Load segmentation results
segmentation_results = load_segmentation_results("E:/segmentation_results.xlsx")

# Streamlit UI
st.title("Blood Vessel Segmentation with UNet Models")

# Select input image
input_image_paths = segmentation_results['Input'].unique()
input_image_labels = [os.path.basename(path) for path in input_image_paths]
input_image_mapping = dict(zip(input_image_labels, input_image_paths))

# Select input image
selected_input_label = st.selectbox("Select Input Image", input_image_labels)
selected_input_image = input_image_mapping[selected_input_label]

# Display selected input image
input_image = Image.open(selected_input_image)
st.image(input_image, caption='Selected Input Image', use_column_width=True)

# Select UNet model
unique_models = pd.unique(segmentation_results['Model'])
selected_model = st.selectbox("Select UNet Model", unique_models)

# Check if the DataFrame is empty or the selected_model is invalid
if not segmentation_results.empty and selected_model in segmentation_results['Model'].values:
    # Filter rows based on the selected_model and selected_input_image
    selected_row = segmentation_results.loc[(segmentation_results['Model'] == selected_model) & 
                                             (segmentation_results['Input'] == selected_input_image)]
    # Further processing with selected_row
    if not selected_row.empty:
        # Get segmentation result and metrics for the selected image and model
        selected_row = selected_row.iloc[0]  # Select the first row if there are multiple matches
        segmentation_image_path = selected_row['Output']
        segmentation_image = Image.open(segmentation_image_path)

        metrics = {
            'Accuracy': selected_row['Accuracy'],
            'IoU_Coefficient': selected_row['IoU_coefficient'],
            'Dice_Coefficient': selected_row['Dice_coefficient'],
            #'F1': selected_row['F1_score'],
            'Precision': selected_row['Precision'],
            'Sensitivity': selected_row['Sensitivity'],
            'Specificity': selected_row['Specificity'],
            'Loss': selected_row['Loss']
        }

        # Display segmentation results and metrics
        display_results(input_image, segmentation_image, metrics)
    else:
        st.write("No segmentation result found for the selected model and input image.")
else:
    st.write("No data found for the selected model or DataFrame is empty.")
