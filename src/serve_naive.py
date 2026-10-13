"""
This file is to serve the UPF classification model as a REST API (for the naive
version).

This is a straightforward API that:
    - Loads the trained model at startup
    - Accepts food item data via a POST request
    - Returns an ultra-processed prediction

This will be improved later with validation, monitoring and better model loading
in later sections
"""
import re
import pickle
from fastapi import FastAPI
from pydantic import BaseModel, Field

ADDITIVE_KEYWORDS = [
    "syrup", "extract", "flavour", "flavor", "hydrogenated", "isolate", "concentrate",
    "emulsifier", "stabilis", "stabiliz", "colorant", "carrageenan", "maltodextrin"
]

CATEGORICAL_COLS = ["top_category", "nutriscore_grade"]

# Load the trained model and encoders at startup
# This is loaded once when the server starts, rather than on every request
print("Loading model")
with open("models/model.pkl", "rb") as f:
    model, encoders = pickle.load(f)

print("Model loaded successfully")

# Create FastAPI application
app = FastAPI(
    title = "UPF Detection API",
    description = """
    Predict whether a food item is ultra-processed (NOVA group 4)

    This API accepts nutrition, ingredient and allergen details and returns:
    - Whether the food is predicted to be ultra-processed
    - The probability of being ultra-processed (0.0 to 1.0)

    Note: This is the naive version without validation or monitoring
    """,
    version = "1.0.0"
)