import os

# Create directories for static files and uploads
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('templates', exist_ok=True)

print("Created required directories:")
print("- static/uploads: for storing user uploaded images")
print("- templates: for HTML templates")
