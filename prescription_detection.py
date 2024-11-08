import cv2
import pytesseract
import numpy as np
from PIL import Image
import logging
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_image(image_path):
    logger.info(f"Preprocessing image: {image_path}")
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise Exception(f"Could not read image: {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        logger.debug("Converted to grayscale")

        # Apply noise reduction
        denoised = cv2.fastNlMeansDenoising(gray)
        logger.debug("Applied noise reduction")

        # Apply thresholding
        _, threshold = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        logger.debug("Applied thresholding")

        # Apply dilation
        kernel = np.ones((2,2), np.uint8)
        dilated = cv2.dilate(threshold, kernel, iterations=1)
        logger.debug("Applied dilation")

        return dilated

    except Exception as e:
        logger.error(f"Error in preprocessing image: {str(e)}")
        raise

def extract_text_from_image(processed_image):
    logger.info("Extracting text from processed image")
    try:
        custom_config = r'--oem 3 --psm 6'
        
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(processed_image)
        
        # Extract text
        text = pytesseract.image_to_string(pil_image, config=custom_config)
        logger.debug(f"Raw extracted text: {text}")
        
        # Clean and format the extracted text
        cleaned_text = clean_extracted_text(text)
        logger.debug(f"Cleaned text: {cleaned_text}")
        
        return cleaned_text if cleaned_text else "No text could be extracted from the image"

    except Exception as e:
        logger.error(f"Error in text extraction: {str(e)}")
        raise

def clean_extracted_text(text):
    logger.info("Cleaning extracted text")
    if not text:
        return ""
    
    # Split text into lines
    lines = text.split('\n')
    
    # Remove empty lines and strip whitespace
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    
    # Join lines back together
    cleaned_text = '\n'.join(cleaned_lines)
    
    return cleaned_text

def process_prescription(image_path):
    logger.info(f"Processing prescription image: {image_path}")
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        processed_image = preprocess_image(image_path)
        extracted_text = extract_text_from_image(processed_image)
        
        logger.info("Prescription processing completed successfully")
        return extracted_text
    except Exception as e:
        logger.error(f"Error processing prescription: {str(e)}")
        return None

if __name__ == "__main__":
    # Replace with the path to your prescription image
    image_path = "path/to/your/prescription_image.jpg"
    
    result = process_prescription(image_path)
    
    if result:
        print("Extracted text from prescription:")
        print(result)
    else:
        print("Failed to process prescription")