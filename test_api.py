import requests
import json
import sys
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def test_object_detection(image_path, api_url='http://localhost:5000/detect'):
    """
    Test the object detection API with the provided image.
    
    Args:
        image_path: Path to the image to test
        api_url: URL of the detection API endpoint
    """
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Prepare the image for sending
    files = {'image': open(image_path, 'rb')}
    
    try:
        # Send the request to the API
        print(f"Sending image to {api_url}...")
        response = requests.post(api_url, files=files)
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            
            # Display the results
            print("\nObjects detected:")
            for obj in result.get('objects', []):
                print(f"- {obj['label']} (confidence: {obj['confidence']})")
            
            # Visualize the results on the image
            visualize_results(image_path, result.get('objects', []))
            
        else:
            print(f"Error: API returned status code {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"Error during API request: {e}")
    
    finally:
        files['image'].close()

def visualize_results(image_path, objects):
    """
    Draw bounding boxes around detected objects and display the image.
    """
    # Open the image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # Draw bounding boxes for each detected object
    for obj in objects:
        x1, y1, x2, y2 = obj['bbox']
        label = f"{obj['label']} ({obj['confidence']})"
        
        # Draw rectangle
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        
        # Draw label
        draw.text((x1, y1-15), label, fill="red")
    
    # Display the image with matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Object Detection Results")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_object_detection(image_path)
    else:
        print("Usage: python test_api.py <path_to_image>")
        print("Example: python test_api.py test_image.jpg")
