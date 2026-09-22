"""
Explore the foods_health_scores_allergens.csv file. 

Check for:
    - Shape
    - Schema/dtypes
    - Missing values
    - Target distribution
    - Duplicates 
    - Nutrition column sanity check
    - Categorical cardinaility
    - Ingredients text
    - Leakage check
    - Allergen flags vs target
"""

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

RAW_PATH = "data/raw/foods_health_scores_allergens.csv"

def main():
    df = pd.read_csv(RAW_PATH)

    # Load data and check shape
    print("-"*60)
    print("SHAPE")
    print("-"*60)
    print(df.shape)
    print(df.columns.tolist())

    # Schema and dtypes 
    print("\n")
    print("-"*60)
    print("DTYPES/INFO")
    print("-"*60)
    df.info()

    # Missing values 
    print("\n")
    print("-"*60)
    print("MISSING VALUES")
    print("-"*60)
    missing = df.isnull().sum().sort_values(ascending=False)
    print(missing[missing>0])

    # Target distribution
    print("\n")
    print("-"*60)
    print("TARGET DISTRIBUTION")
    print("-"*60)
    print(df["nova_group"].value_counts(dropna=False))

    labelled = df.dropna(subset=["nova_group"]).copy()
    labelled["is_ultra_processed"] = (labelled["nova_group"] == 4).astype(int)
    print("\nProcessed food balance: ")
    print(labelled["is_ultra_processed"].value_counts(normalize=True))

    # Duplicates 
    print("\n")
    print("-"*60)
    print("DUPLICATES")
    print("-"*60)
    print("Full-row duplicates: ", df.duplicated().sum())
    print("Duplicate product name and brand pairs: ", df.duplicated(subset=["product_name", "brands"]).sum())

    # Nutrition column sanity check
    print("\n")
    print("-"*60)
    print("NUTRITION COLUMN STATS")
    print("-"*60)
    nutrition_cols = ["energy_kcal", "fat_100g", "carbs_100g", "sugars_100g",
                      "proteins_100g", "fiber_100g", "salt_100g", "sodium_100g"]
    print(df[nutrition_cols].describe())

    # Categorical cardinality
    print("\n")
    print("-"*60)
    print("CATEGORICAL CARDINALITY")
    print("-"*60)
    for col in ["categories", "brands", "nutriscore_grade", "ecoscore_grade"]:
        if col in df.columns:
            print(f"{col}: {df[col].nunique()} unique values")
            print(df[col].value_counts(dropna=False).head(10))
            print()

    # Ingredients text
    print("\n")
    print("-"*60)
    print("INGREDIENTS TEXT")
    print("-"*60)
    if "ingredients" in df.columns:
        ing_len = df["ingredients"].dropna()
        ing_commas = df["ingredients"].dropna().str.count(",")
        print("ingredient string length:\n", ing_len.describe())
        print("\ningredient comma count(proxy for number of ingredients):\n", ing_commas.describe())

        print("\nSample NOVA 4 ingredients:")
        for txt in labelled.loc[labelled["nova_group"] == 4, "ingredients"].dropna().head(5):
            print("-", txt[:200])

        print("\nSample NOVA 1 ingredients:")
        for txt in labelled.loc[labelled["nova_group"] == 1, "ingredients"].dropna().head(5):
            print("-", txt[:200])

    # Leakage check
    print("\n")
    print("-"*60)
    print("NUTRISCORE/ECOSCORE VS NOVA GROUP")
    print("-"*60)

    if "nutriscore_grade" in df.columns:
        print(pd.crosstab(df["nutriscore_grade"], df["nova_group"], normalize="index"))

    if "ecoscore_grade" in df.columns:
        print(pd.crosstab(df["ecoscore_grade"], df["nova_group"], normalize="index"))

    # Allergen flags vs target
    print("\n")
    print("-"*60)
    print("ALLERGEN FLAGS VS is_ultra_processed")
    print("-"*60)
    allergen_cols = [
        "contains_gluten", "contains_dairy", "contains_nuts", "contains_soy", "contains_eggs",
        "contains_fish"
    ] 
    print(labelled.groupby("is_ultra_processed")[allergen_cols].mean())

if __name__ == "__main__":
    main()

# Notes:
# ------
# Shape is (4997, 24)
# All variables are object, float64 and bool
# Most missing values are allergens and fiber. Allergens missing isn't concerning
# but fiber is strange. Lots of nova group missing which could be an issue. 17
# columns contain missing data out of 24
# Imbalanced target distribution. Most have nova score of 4.0 (2637/4997). Next
# most populous group is 3.0 (1093/4997). After creating is_ultraa_processed target
# column, UPF food is 58.3% of the food items. 
# There are 38 full row duplicates and 218 duplicate product name and brand pairs
# There is something iffy going on with the fiber column. The mean is 1.7 billion and
# the max is 6.1 trillion for a per-100g nutrient. This is clearly nonsense and needs 
# to be dealt with. Probably bad units or some kind of encoding error but should dig
# into this later. A few others like carbs per 100g have crazy maxes like 8989 so some
# further investigation clearly needs to happen here. 
# A few notes about categorical cardinality: categories (430 values) and brands (2063 values)
# are both too high cardinality for direct one-hot encoding. Will also need to be investigated/
# dealt with before modelling. nutriscore_grade/ecoscore_grade aren't clear A-E either
# because they both have UNKNOWN/NOT-APPLICABLE/NaN as separate values. Need to determine whether
# these count as missing and should be imputed or a legitimate category in its own right.
# Also should NaN among the brand categories be dealt with as it's own category? I would assume
# these are whole foods. 
# The NOVA 4 ingredient list is very long in comparison with NOVA 1, which is to be expected. 
# There are lots of additive/processing markers eg stabilisants, colorant e150d, sirop de glucose,
# etc. NOVA 1 is nearly all single ingredient entries 
# The nova_group and nutriscore_grade crosstabulation shows the NOVA 4 share growing steadily
# from 34% at grade A to 75% at grade E, so it's correlated with ultra-processing but not redundant
# as A grade foods are still 34% NOVA 4. ecoscore_grade doesn't have the same steady pattern, no 
# monotonic trend to be seen. NOVA 4 bounces between 39-66% across A-F. It's tracking environmental
# impact not level of processing so it's weak/noisy as a predictor compared to nutriscore_grade.
# Should likely deprioritise rather than treating as a leakage risk. 
# From the allergen flags vs ultraprocessed crosstabulation, allergens are much more present 
# most allergens are far more common in UPFs. Gluten is 24% -> 42%, soy is 5% -> 28%. Nuts and eggs also. 
# All roughly doubled or more. Only contains_fish is more common in non-UPF food which makes sense as fish 
# is more likely to be present in minimally processed food than UPFs. Could be useful features, soy in 
# particular.