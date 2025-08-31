# fix_images.py - Download and save product images locally
import requests
import os
from PIL import Image
from io import BytesIO
from database import SessionLocal
import models
import crud

def download_and_save_images():
    print("Downloading and saving product images locally...")
    
    # Create static/sample_images directory
    static_images_dir = os.path.join('static', 'sample_images')
    os.makedirs(static_images_dir, exist_ok=True)
    
    db = SessionLocal()
    
    try:
        products = db.query(models.Product).all()
        
        for product in products:
            if product.image_path.startswith('http'):
                try:
                    print(f"Downloading image for {product.product_id}...")
                    
                    # Download image
                    response = requests.get(product.image_path)
                    response.raise_for_status()
                    
                    # Open and save image
                    image = Image.open(BytesIO(response.content))
                    
                    # Create filename
                    filename = f"{product.product_id}.jpg"
                    local_path = os.path.join(static_images_dir, filename)
                    
                    # Save image
                    image.save(local_path)
                    
                    # Update database with local path
                    product.image_path = f"sample_images/{filename}"
                    db.commit()
                    
                    print(f"  ✅ Saved {filename}")
                    
                except Exception as e:
                    print(f"  ❌ Error downloading {product.product_id}: {e}")
                    continue
        
        print("Image download completed!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    download_and_save_images()
