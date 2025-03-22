from flask import Flask, request, jsonify, render_template, redirect, url_for
from object_detector import ObjectDetector
import os
import uuid
import base64
from io import BytesIO
from PIL import Image, ImageDraw

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the model on first request to avoid loading on startup
detector = None

def get_detector():
    global detector
    if detector is None:
        from object_detector import ObjectDetector
        detector = ObjectDetector()
    return detector

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_objects():
    if request.content_type and 'application/json' in request.content_type:
        # API endpoint for JSON requests
        return jsonify({'error': 'Direct JSON upload not supported yet'}), 400
    
    # Form submission from the web interface
    if 'image' not in request.files:
        return render_template('index.html', error='No image provided')
    
    image_file = request.files['image']
    if image_file.filename == '':
        return render_template('index.html', error='No image selected')
    
    # Get the model
    detector = get_detector()
    
    # Generate a unique filename
    filename = str(uuid.uuid4()) + os.path.splitext(image_file.filename)[1]
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Save the uploaded image
    image_file.save(filepath)
    
    # Detect objects in the image
    results = detector.detect(filepath)
    
    # Draw bounding boxes on image
    img = Image.open(filepath)
    draw = ImageDraw.Draw(img)
    
    for obj in results:
        x1, y1, x2, y2 = obj['bbox']
        label = f"{obj['label']} ({obj['confidence']})"
        # Draw rectangle
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        # Draw label
        draw.text((x1, y1-15), label, fill="red")
    
    # Save the annotated image
    annotated_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"annotated_{filename}")
    img.save(annotated_filepath)
    
    return render_template('results.html', 
                          original_image=f'/static/uploads/{filename}',
                          annotated_image=f'/static/uploads/annotated_{filename}',
                          results=results)

@app.route('/api/detect', methods=['POST'])
def api_detect_objects():
    # Original API endpoint for programmatic access
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Get the model
    detector = get_detector()
    
    # Save the uploaded image temporarily
    temp_path = 'temp_image.jpg'
    image_file.save(temp_path)
    
    # Detect objects in the image
    results = detector.detect(temp_path)
    
    # Clean up the temporary file
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return jsonify({'objects': results})

if __name__ == '__main__':
    # Use environment variable for port if available (for Render)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
