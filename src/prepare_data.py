"""
Prepare the UPF classification dataset from the raw Open Foof Facts dataset.
Cleans, engineers features, derives target and writes data/train.csv and 
data/test.csv.
"""
import numpy as np
import pandas as od
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/foods_health_scores_allergens.csv"

NUTRITION_COLS = [
    "energy_kcal", "fat_100g, saturated_fat_100g", "carbs_100g", "sugars_100g",
    "fiber_100g", "proteins_100g", "salt_100g", "sodium_100g"
]

# per-100g nutrients can't possibly exceed 100g of product
# The most calorie dense macro that exists is fat, around 9 calories per gram
# So 100g of pure fat would be 900 calories. No real food is more calorie dense
# than that so cap it here.
NUTRITION_MAX = {col: 100 for col in NUTRITION_COLS}
NUTRITION_MAX["energy_kcal"] = 900

ADDITIVE_KEYWORDS = [
    "syrup", "extract", "flavour", "flavor", "hydrogenated", "isolate", "concentrate",
    "emulsifier", "stabilis", "stabiliz", "colorant", "carrageenan", "maltodextrin"
]

ALLERGEN_COLS = [
    "contains_gluten", "contains_dairy", "contains_nuts", "contains_soy", "contains_eggs",
    "contains_fish"
]

def add_ingredient_features(df):
    """
    Create the following:
        - ingredient_count: a proxy for how many ingredients a food has. NOVA 4
          should have more in theory.
        - has_e_number: regex searched for patterns like "E150" and "E621" that
          are known to be present in NOVA 4 sample ingredients
        - has_additive_keyword: checks for any of the words in ADDITIVE_KEYWORDS
    """
    ingredients = df["ingredients"]
    df["ingredient_count"] = ingredients.str.count(",") + 1
    df["has_e_number"] = ingredients.str.contains(r"\be\d{3}\b", case = False, regex = True)
    df["has_additive_keyword"] = ingredients.str.contains(
        "|".join(ADDITIVE_KEYWORDS), case = False, regex = True
    )
    return df

def clean_nutrition(df):
    """
    Converts corrupted values to NaNs and fills every NaN in each nutrition column
    with that column's median so rows don't get dropped because one nutrient reading
    was bad or missing.

    This form of median imputation is just a starting point for the naive model. Can
    tweak and improve later.
    """
    for col, max_val in NUTRITION_MAX.items():
        df.loc[(df[col] < 0) | (df[col] > max_val), col] = np.nan

    for col in NUTRITION_COLS:
        df[col] = df[col].fillna(df[col].median())

    return df

def simplify_category(df):
    pass

def clean_nutriscore(df):
    pass

def main():
    pass

if __name__ == "__main__":
    main()