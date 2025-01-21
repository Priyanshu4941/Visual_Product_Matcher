from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.models as vision_models  # Changed from models to vision_models
import requests
from io import BytesIO
from database import SessionLocal
import models
import pickle
import shutil
# Add these imports
import uuid
from werkzeug.utils import secure_filename
import os
from collections import defaultdict
import re

# Add configurations
UPLOAD_FOLDER = os.path.join('app', 'uploads')

# Add these configurations after UPLOAD_FOLDER
STATIC_UPLOADS = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}


# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_UPLOADS, exist_ok=True)


app = Flask(__name__)

# Add this helper function
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize model and transformations
def get_model():
    model = vision_models.resnet50(pretrained=True)  # Using vision_models
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    return model

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def extract_features(image):
    """Extract features from an uploaded image"""
    model = get_model()
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        features = model(image_tensor)
    
    return features.squeeze().numpy()

def ensure_unique_results(results, similarity_threshold=0.95):
    """Ensure results are unique based on feature similarity"""
    unique_results = []
    seen_features = set()
    
    for result in results:
        feature_hash = hash(tuple(pickle.loads(result['feature_vector']).flatten()))
        
        # Check if this feature is significantly different from seen features
        is_unique = True
        for seen_hash in seen_features:
            if abs(feature_hash - seen_hash) / max(abs(feature_hash), abs(seen_hash)) < similarity_threshold:
                is_unique = False
                break
        
        if is_unique:
            seen_features.add(feature_hash)
            unique_results.append(result)
    
    return unique_results

def find_similar_products(query_features, num_results=8):
    """Find similar products using the database"""
    db = SessionLocal()
    try:
        products = db.query(models.Product).all()
        print(f"Total products in database: {len(products)}")
        
        similarities = []
        seen_features = set()  # Track unique feature vectors
        
        for product in products:
            try:
                stored_features = pickle.loads(product.feature_vector)
                
                # Create a feature hash for comparison
                feature_hash = hash(tuple(stored_features.flatten()))
                
                # Skip if we've seen very similar features
                if feature_hash in seen_features:
                    continue
                
                similarity = np.dot(stored_features, query_features) / (
                    np.linalg.norm(stored_features) * np.linalg.norm(query_features)
                )
                
                # Only add if similarity is above threshold
                if similarity > 0.1:  # Adjust threshold as needed
                    seen_features.add(feature_hash)
                    
                    similarities.append({
                        'path': product.image_path,
                        'category': product.category,
                        'similarity': similarity,
                        'feature_hash': feature_hash
                    })
                    print(f"Added unique product: {product.image_path} with similarity: {similarity:.2f}")
            
            except Exception as e:
                print(f"Error processing product: {str(e)}")
                continue
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Take top results ensuring uniqueness
        top_results = []
        seen_hashes = set()
        
        for result in similarities:
            if len(top_results) >= num_results:
                break
            
            if result['feature_hash'] not in seen_hashes:
                seen_hashes.add(result['feature_hash'])
                top_results.append({
                    'path': result['path'],
                    'category': result['category'],
                    'similarity': result['similarity']
                })
                print(f"Selected for display: {result['path']}")
        
        print(f"\nFinal unique results: {len(top_results)}")
        return top_results
    
    except Exception as e:
        print(f"Error in find_similar_products: {str(e)}")
        return []
    finally:
        db.close()

@app.route('/')
def index():
    return render_template('index.html', images=None)

@app.route('/', methods=['POST'])
def upload_file():
    try:
        image = None
        saved_path = None
        preview_image = None
        print("Upload request received")
        
        # Ensure folders exist
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)
        
        if 'file' in request.files:
            # Handle file upload
            file = request.files['file']
            if file and file.filename != '':
                if allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    unique_filename = f"{uuid.uuid4()}_{filename}"
                    saved_path = os.path.join(UPLOAD_FOLDER, unique_filename)
                    
                    file.save(saved_path)
                    print(f"File saved to: {saved_path}")
                    
                    image = Image.open(saved_path).convert('RGB')
                    
                    static_path = os.path.join('static', 'uploads', unique_filename)
                    image.save(static_path)
                    
                    preview_image = f"uploads/{unique_filename}"
                    print("File uploaded and processed successfully")
                else:
                    return render_template('index.html', error="Invalid file type")
                    
        elif 'imageUrl' in request.form:
            url = request.form['imageUrl'].strip()
            if url:
                try:
                    print(f"Downloading image from URL: {url}")
                    
                    # Add headers to mimic a browser request
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    }
                    
                    # Download image with timeout and stream options
                    response = requests.get(
                        url, 
                        headers=headers, 
                        timeout=15, 
                        stream=True,
                        verify=False  # Only if needed for SSL issues
                    )
                    response.raise_for_status()
                    
                    # Get content type and extension
                    content_type = response.headers.get('content-type', '').lower()
                    
                    # Determine file extension
                    if 'jpeg' in content_type or 'jpg' in content_type:
                        ext = '.jpg'
                    elif 'png' in content_type:
                        ext = '.png'
                    elif 'webp' in content_type:
                        ext = '.webp'
                    else:
                        return render_template('index.html', error="Unsupported image format")
                    
                    # Generate unique filename
                    unique_filename = f"url_image_{uuid.uuid4()}{ext}"
                    saved_path = os.path.join(UPLOAD_FOLDER, unique_filename)
                    static_path = os.path.join('static', 'uploads', unique_filename)
                    
                    # Save the downloaded image
                    with open(saved_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    # Open and verify the image
                    try:
                        image = Image.open(saved_path)
                        # Convert to RGB if necessary
                        if image.mode in ('RGBA', 'P'):
                            image = image.convert('RGB')
                        
                        # Save a copy to static folder
                        image.save(static_path)
                        
                        preview_image = f"uploads/{unique_filename}"
                        print(f"URL image saved successfully: {saved_path}")
                        
                    except Exception as e:
                        if os.path.exists(saved_path):
                            os.remove(saved_path)
                        if os.path.exists(static_path):
                            os.remove(static_path)
                        raise Exception(f"Invalid image file: {str(e)}")
                    
                except requests.exceptions.RequestException as e:
                    error_msg = f"Failed to download image: {str(e)}"
                    print(error_msg)
                    return render_template('index.html', error=error_msg)
                except Exception as e:
                    error_msg = f"Error processing URL image: {str(e)}"
                    print(error_msg)
                    return render_template('index.html', error=error_msg)
            else:
                return render_template('index.html', error="Please provide a valid image URL")
        
        if image is None:
            return render_template('index.html', error="No valid image provided")

        # Extract features
        print("Extracting features...")
        query_features = extract_features(image)
        print("Features extracted successfully")

        # Find similar products
        print("Finding similar products...")
        similar_products = find_similar_products(query_features)
        print(f"Found {len(similar_products)} similar products")

        # Format results
        formatted_products = []
        seen_paths = set()
        
        for product in similar_products:
            if product['path'] not in seen_paths:
                seen_paths.add(product['path'])
                formatted_products.append({
                    'path': product['path'],
                    'category': product['category'],
                    'similarity': f"{float(product['similarity']):.2f}"
                })

        print(f"Formatted {len(formatted_products)} unique products")

        # Clean up old files
        cleanup_old_uploads()

        return render_template('index.html', 
                             images=formatted_products,
                             preview_image=preview_image)

    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template('index.html', error=str(e))

# Add helper function to validate URLs
def is_valid_image_url(url):
    """Check if the URL points to a valid image"""
    try:
        response = requests.head(url, timeout=5)
        content_type = response.headers.get('content-type', '').lower()
        return any(img_type in content_type for img_type in ['jpeg', 'jpg', 'png', 'webp'])
    except:
        return False

# Add function to handle different image formats
def process_image(image):
    """Process image to ensure it's in the correct format"""
    if image.mode in ('RGBA', 'P'):
        return image.convert('RGB')
    return image

# Add error handling middleware
@app.errorhandler(413)  # Request Entity Too Large
def request_entity_too_large(error):
    return render_template('index.html', error="File too large"), 413

@app.errorhandler(500)  # Internal Server Error
def internal_server_error(error):
    return render_template('index.html', error="Internal server error"), 500
    

def cleanup_old_uploads(max_files=50):
    """Clean up old uploaded files, keeping only the most recent ones"""
    try:
        # Clean uploads folder
        upload_files = []
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                upload_files.append((file_path, os.path.getmtime(file_path)))
        
        # Clean static/uploads folder
        static_upload_folder = os.path.join('static', 'uploads')
        static_files = []
        if os.path.exists(static_upload_folder):
            for filename in os.listdir(static_upload_folder):
                file_path = os.path.join(static_upload_folder, filename)
                if os.path.isfile(file_path):
                    static_files.append((file_path, os.path.getmtime(file_path)))
        
        # Sort files by modification time
        upload_files.sort(key=lambda x: x[1], reverse=True)
        static_files.sort(key=lambda x: x[1], reverse=True)
        
        # Remove old files
        for file_path, _ in upload_files[max_files:]:
            try:
                os.remove(file_path)
                print(f"Cleaned up old file: {file_path}")
            except Exception as e:
                print(f"Error removing file {file_path}: {str(e)}")
                
        for file_path, _ in static_files[max_files:]:
            try:
                os.remove(file_path)
                print(f"Cleaned up old static file: {file_path}")
            except Exception as e:
                print(f"Error removing static file {file_path}: {str(e)}")
                
    except Exception as e:
        print(f"Error in cleanup_old_uploads: {str(e)}")



    
# @app.route('/check_duplicates')
# def check_duplicates():
#     db = SessionLocal()
#     try:
#         products = db.query(models.Product).all()
        
#         # Track duplicates
#         path_counts = {}
#         filename_counts = {}
        
#         for product in products:
#             # Count by full path
#             path = product.image_path
#             path_counts[path] = path_counts.get(path, 0) + 1
            
#             # Count by filename
#             filename = os.path.basename(path)
#             filename_counts[filename] = filename_counts.get(filename, 0) + 1
        
#         # Find duplicates
#         duplicate_paths = {k: v for k, v in path_counts.items() if v > 1}
#         duplicate_files = {k: v for k, v in filename_counts.items() if v > 1}
        
#         return jsonify({
#             'total_products': len(products),
#             'unique_paths': len(path_counts),
#             'unique_filenames': len(filename_counts),
#             'duplicate_paths': duplicate_paths,
#             'duplicate_files': duplicate_files
#         })
        
#     finally:
#         db.close()

# @app.route('/remove_duplicates')
# def remove_duplicates():
#     db = SessionLocal()
#     try:
#         products = db.query(models.Product).all()
        
#         # Track unique products by feature hash
#         unique_products = {}
#         removed = 0
        
#         for product in products:
#             try:
#                 stored_features = pickle.loads(product.feature_vector)
#                 feature_hash = hash(tuple(stored_features.flatten()))
                
#                 if feature_hash not in unique_products:
#                     unique_products[feature_hash] = product
#                 else:
#                     db.delete(product)
#                     removed += 1
#             except Exception as e:
#                 print(f"Error processing product: {str(e)}")
#                 continue
        
#         db.commit()
        
#         return jsonify({
#             'success': True,
#             'total_before': len(products),
#             'total_after': len(unique_products),
#             'removed': removed
#         })
        
#     except Exception as e:
#         db.rollback()
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         })
#     finally:
#         db.close()

# @app.route('/debug_template')
# def debug_template():
#     """Debug route to check template rendering"""
#     db = SessionLocal()
#     try:
#         product = db.query(models.Product).first()
#         if product:
#             test_data = [{
#                 'path': product.image_path,
#                 'category': product.category,
#                 'similarity': '1.00'
#             }]
#             print("Template data:", test_data)
#             return render_template('index.html', images=test_data)
#         return "No products found"
#     finally:
#         db.close()

if __name__ == '__main__':
    app.run(debug=True)