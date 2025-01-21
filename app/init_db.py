# init_db.py
import os
import numpy as np
from database import engine, SessionLocal
import models
from models import Base
import crud
from tqdm import tqdm

def init_database():
    print("Starting database initialization...")
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    try:
        # Load features and metadata
        features_path = os.path.join('models', 'features.npy')
        image_paths_file = os.path.join('models', 'image_paths.txt')
        categories_file = os.path.join('models', 'categories.txt')
        
        # Load feature vectors
        print("Loading feature vectors...")
        feature_vectors = np.load(features_path)
        print(f"Loaded {len(feature_vectors)} feature vectors")
        
        # Load image paths
        print("Loading image paths...")
        with open(image_paths_file, 'r') as f:
            image_paths = f.read().splitlines()
            
        # Load categories
        print("Loading categories...")
        with open(categories_file, 'r') as f:
            categories = f.read().splitlines()
        
        # Verify all lists have the same length
        if not (len(feature_vectors) == len(image_paths) == len(categories)):
            raise ValueError(
                f"Mismatch in lengths: features({len(feature_vectors)}), "
                f"paths({len(image_paths)}), categories({len(categories)})"
            )
            
        # Create database session
        db = SessionLocal()
        
        print("\nPopulating database...")
        # Use tqdm for progress bar
        for idx in tqdm(range(len(feature_vectors))):
            # Get product ID from image path
            product_id = os.path.splitext(os.path.basename(image_paths[idx]))[0]
            
            # Check if product already exists
            existing_product = crud.get_product(db, product_id)
            if not existing_product:
                # Create new product
                crud.create_product(
                    db=db,
                    product_id=product_id,
                    category=categories[idx],
                    image_path=image_paths[idx],
                    feature_vector=feature_vectors[idx]
                )
        
        print("\nDatabase initialization completed successfully")
        print(f"Total products added: {len(feature_vectors)}")
        
        # Print category distribution
        print("\nCategory distribution:")
        category_counts = {}
        for category in categories:
            category_counts[category] = category_counts.get(category, 0) + 1
        for category, count in category_counts.items():
            print(f"{category}: {count} products")
            
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    # Install tqdm if not already installed
    try:
        from tqdm import tqdm
    except ImportError:
        print("Installing tqdm for progress bar...")
        os.system('pip install tqdm')
        from tqdm import tqdm
    
    init_database()