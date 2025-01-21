# models.py - Database model definition
from sqlalchemy import Column, Integer, String, Float, LargeBinary
from database import Base

class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(String, unique=True, index=True)
    category = Column(String, index=True)
    image_path = Column(String)
    feature_vector = Column(LargeBinary)