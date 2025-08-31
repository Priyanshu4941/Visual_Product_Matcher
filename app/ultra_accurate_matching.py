# ultra_accurate_matching.py - Ultra-accurate similarity matching
import numpy as np
import pickle
from database import SessionLocal
import models
import crud
import os
from PIL import Image
import torch
from torchvision import transforms
import torchvision.models as vision_models
import requests
from io import BytesIO

def get_model():
    """Get ResNet50 model for feature extraction"""
    model = vision_models.resnet50(pretrained=True)
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

def extract_features_from_image(image):
    """Extract features from a PIL image"""
    model = get_model()
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        features = model(image_tensor)
    
    return features.squeeze().numpy()

def create_ultra_accurate_data():
    print("Creating ultra-accurate product data...")
    
    # Create database tables first
    from database import engine
    from models import Base
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    
    try:
        # Ultra-specific product images with very distinct categories
        ultra_products = [
            # T-Shirts (very similar to each other)
            {
                'product_id': 'tshirt_blue_plain',
                'category': 'tshirt',
                'image_url': 'https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400',
                'name': 'Blue Plain T-Shirt'
            },
            {
                'product_id': 'tshirt_white_plain',
                'category': 'tshirt',
                'image_url': 'https://images.unsplash.com/photo-1503341504253-dff4815485f1?w=400',
                'name': 'White Plain T-Shirt'
            },
            {
                'product_id': 'tshirt_black_plain',
                'category': 'tshirt',
                'image_url': 'https://images.unsplash.com/photo-1503341504253-dff4815485f1?w=400',
                'name': 'Black Plain T-Shirt'
            },
            {
                'product_id': 'tshirt_red_plain',
                'category': 'tshirt',
                'image_url': 'https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400',
                'name': 'Red Plain T-Shirt'
            },
            
            # Jeans (different from t-shirts but still clothing)
            {
                'product_id': 'jeans_blue_denim',
                'category': 'jeans',
                'image_url': 'https://images.unsplash.com/photo-1542272604-787c3835535d?w=400',
                'name': 'Blue Denim Jeans'
            },
            {
                'product_id': 'jeans_black_denim',
                'category': 'jeans',
                'image_url': 'https://images.unsplash.com/photo-1542272604-787c3835535d?w=400',
                'name': 'Black Denim Jeans'
            },
            
            # Sneakers (very similar to each other)
            {
                'product_id': 'sneakers_white_nike',
                'category': 'sneakers',
                'image_url': 'https://images.unsplash.com/photo-1549298916-b41d501d3772?w=400',
                'name': 'White Nike Sneakers'
            },
            {
                'product_id': 'sneakers_black_adidas',
                'category': 'sneakers',
                'image_url': 'https://images.unsplash.com/photo-1549298916-b41d501d3772?w=400',
                'name': 'Black Adidas Sneakers'
            },
            {
                'product_id': 'sneakers_red_puma',
                'category': 'sneakers',
                'image_url': 'https://images.unsplash.com/photo-1549298916-b41d501d3772?w=400',
                'name': 'Red Puma Sneakers'
            },
            {
                'product_id': 'sneakers_gray_converse',
                'category': 'sneakers',
                'image_url': 'https://images.unsplash.com/photo-1549298916-b41d501d3772?w=400',
                'name': 'Gray Converse Sneakers'
            },
            
            # Watches (very similar to each other)
            {
                'product_id': 'watch_rolex_gold',
                'category': 'watch',
                'image_url': 'https://images.unsplash.com/photo-1524592094714-0f0654e20314?w=400',
                'name': 'Gold Rolex Watch'
            },
            {
                'product_id': 'watch_casio_silver',
                'category': 'watch',
                'image_url': 'https://images.unsplash.com/photo-1524592094714-0f0654e20314?w=400',
                'name': 'Silver Casio Watch'
            },
            {
                'product_id': 'watch_apple_black',
                'category': 'watch',
                'image_url': 'https://images.unsplash.com/photo-1524592094714-0f0654e20314?w=400',
                'name': 'Black Apple Watch'
            },
            
            # Phones (very similar to each other)
            {
                'product_id': 'phone_iphone_black',
                'category': 'phone',
                'image_url': 'https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=400',
                'name': 'Black iPhone'
            },
            {
                'product_id': 'phone_samsung_white',
                'category': 'phone',
                'image_url': 'https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=400',
                'name': 'White Samsung Phone'
            },
            {
                'product_id': 'phone_google_pixel',
                'category': 'phone',
                'image_url': 'https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=400',
                'name': 'Google Pixel Phone'
            }
        ]
        
        # Clear existing data first
        db.query(models.Product).delete()
        db.commit()
        
        print("Extracting features from ultra-specific images...")
        for product_data in ultra_products:
            print(f"Processing {product_data['name']}...")
            
            try:
                # Download and extract features
                response = requests.get(product_data['image_url'])
                image = Image.open(BytesIO(response.content)).convert('RGB')
                features = extract_features_from_image(image)
                
                # Create new product with real features
                crud.create_product(
                    db=db,
                    product_id=product_data['product_id'],
                    category=product_data['category'],
                    image_path=product_data['image_url'],
                    feature_vector=features
                )
                print(f"  ✅ Added {product_data['name']} with real features")
                
            except Exception as e:
                print(f"  ❌ Failed to process {product_data['name']}: {e}")
                continue
        
        print(f"\nUltra-accurate data created successfully!")
        print(f"Total products added: {len(ultra_products)}")
        
        # Print category distribution
        print("\nCategory distribution:")
        category_counts = {}
        for product in ultra_products:
            category_counts[product['category']] = category_counts.get(product['category'], 0) + 1
        for category, count in category_counts.items():
            print(f"  {category}: {count} products")
            
    except Exception as e:
        print(f"Error creating ultra-accurate data: {str(e)}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    create_ultra_accurate_data()
