"""
Prepare the UPF classification dataset from the raw Open Foof Facts dataset.
Cleans, engineers features, derives target and writes data/train.csv and 
data/test.csv.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/foods_health_scores_allergens.csv"

NUTRITION_COLS = [
    "energy_kcal", "fat_100g", "saturated_fat_100g", "carbs_100g", "sugars_100g",
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
    """
    Creates a new top_category column with fewer unique values than the original 430.
    Makes encoding possible.

    Fills missing values with the literal string "unknown", splits on "," and takes first
    element (which is a broad general tag) then strips any leading or tailing whitespace
    left after the split
    """
    df["top_category"] = df["categories"].fillna("unknown").str.split(",").str[0].str.strip()
    return df

def clean_nutriscore(df):
    """
    Removes nutriscore_grade column (EDA shows they have UNKNOWN, NOT-APPLICABLE and NaN) and collapses them 
    into one consistent label ("unknown")
    """
    df["nutriscore_grade"] = (
        df["nutriscore_grade"].replace({"UNKNOWN": "unknown", "NOT-APPLICABLE": "unknown"}).fillna("unknown")
    )

    return df

def main():
    df = pd.read_csv(RAW_PATH)

    # Only rows with a known NOVA group can be labelled
    df = df.dropna(subset=["nova_group"]).copy()
    df["is_ultra_processed"] = (df["nova_group"] == 4).astype(int)

    # Duplicates: exact rows, then repeated product_name and brand pairs
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["product_name", "brands"])

    # Ingredients text drives the key engineered features so drop rows missing it
    df = df.dropna(subset=["ingredients"])

    df = clean_nutrition(df)
    df = add_ingredient_features(df)
    df = simplify_category(df)
    df = clean_nutriscore(df)
    df[ALLERGEN_COLS] = df[ALLERGEN_COLS].astype(int)

    feature_cols = (
        NUTRITION_COLS + ALLERGEN_COLS
        + ["ingredient_count", "has_e_number", "has_additive_keyword", "top_category", "nutriscore_grade"]
    )
    df = df[feature_cols + ["is_ultra_processed"]]

    train_df, test_df = train_test_split(
        df, test_size = 0.2, stratify = df["is_ultra_processed"], random_state = 42
    )

    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print("Dataset prepared")
    print(f"Training set: {len(train_df):,} food items")
    print(f"Test set: {len(test_df):,} food items")
    print(f"Overall UPF ratio: {df["is_ultra_processed"].mean():.2%}")

    non_upf_sugar = df.loc[df["is_ultra_processed"] == 0, "sugars_100g"].mean()
    upf_sugar = df.loc[df["is_ultra_processed"] == 1, "sugars_100g"].mean()
    print(f"\nNon-UPF items. Average sugars per 100g: {non_upf_sugar:.2f}g")
    print(f"UPF items. Average sugars per 100g: {upf_sugar:.2f}g")

    print("\nTop category distribution (UPF): ")
    print(df.loc[df["is_ultra_processed"] == 1, "top_category"].value_counts(normalize=True).head(5))

if __name__ == "__main__":
    main()