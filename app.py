from flask import Flask, render_template, request, flash
from prescription_detection import preprocess_image, extract_text_from_image
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for flash messages

# Create static folder if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            flash('No file uploaded')
            return render_template('index.html')

        file = request.files['image']
        if file.filename == '':
            flash('No file selected')
            return render_template('index.html')

        # Check file extension
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if not file.filename.lower().endswith(tuple(allowed_extensions)):
            flash('Invalid file type. Please upload an image file (PNG, JPG, JPEG)')
            return render_template('index.html')

        # Save uploaded file
        image_path = os.path.join("static", "uploaded_prescription.jpg")
        file.save(image_path)
        logger.debug(f"Image saved to {image_path}")

        # Preprocess and analyze
        processed_image = preprocess_image(image_path)
        logger.debug("Image preprocessing completed")

        extracted_text = extract_text_from_image(processed_image)
        logger.debug(f"Extracted text: {extracted_text}")

        return render_template('result.html', 
                             text=extracted_text, 
                             image_path=image_path)

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        flash(f"An error occurred: {str(e)}")
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)