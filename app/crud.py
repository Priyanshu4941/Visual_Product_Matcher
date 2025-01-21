# crud.py - Database operations
from sqlalchemy.orm import Session
import numpy as np
from models import Product
import pickle

def create_product(db: Session, product_id: str, category: str, image_path: str, feature_vector: np.ndarray):
    db_product = Product(
        product_id=product_id,
        category=category,
        image_path=image_path,
        feature_vector=pickle.dumps(feature_vector)
    )
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    return db_product

def get_product(db: Session, product_id: str):
    return db.query(Product).filter(Product.product_id == product_id).first()