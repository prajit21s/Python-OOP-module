import pandas as pd
import numpy as np

# 1. READ THE CSV FILE


df = pd.read_csv("pandas_dataset.csv")

# Save original shape
original_shape = df.shape

print("=" * 60)
print("1. FIRST 5 ROWS OF ORIGINAL DATASET")
print("=" * 60)
print(df.head())


# 2. RENAME / CLEAN COLUMN NAMES
# Remove spaces, convert to lowercase, replace spaces with underscores

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\n" + "=" * 60)
print("2. CLEANED COLUMN NAMES")
print("=" * 60)
print(df.columns.tolist())


# 3. DROP UNNECESSARY BLANK ROWS

df = df.dropna(how="all")

# Reset index
df = df.reset_index(drop=True)

print("\n" + "=" * 60)
print("3. SHAPE AFTER REMOVING BLANK ROWS")
print("=" * 60)
print(df.shape)


# 4. DROP COLUMNS IF NEEDED

# No columns are dropped because all columns are useful
# for the analysis.



# 5. COUNT MISSING VALUES

print("\n" + "=" * 60)
print("5. MISSING VALUES IN EACH COLUMN")
print("=" * 60)

print(df.isnull().sum())


# 6A. DROP ROWS WHERE SALES IS MISSING

df = df.dropna(subset=["sales"])

# Reset index
df = df.reset_index(drop=True)

print("\n" + "=" * 60)
print("6A. SHAPE AFTER DROPPING MISSING SALES")
print("=" * 60)
print(df.shape)


# 6B. FILL MISSING PROFIT USING MEAN

profit_mean = df["profit"].mean()

print("\nMean Profit:")
print(profit_mean)

df["profit"] = df["profit"].fillna(profit_mean)


# 6C. FILL MISSING DISCOUNT USING MEDIAN


discount_median = df["discount"].median()

print("\nMedian Discount:")
print(discount_median)

df["discount"] = df["discount"].fillna(discount_median)


# 7. HANDLE MISSING CUSTOMER NAME

df["customer_name"] = df["customer_name"].fillna("Unknown")

print("\n" + "=" * 60)
print("7. CUSTOMER NAMES AFTER FILLING MISSING VALUES")
print("=" * 60)

print(df[["order_id", "customer_name"]])


# 8. DETECT DUPLICATES BASED ON ORDER_ID

print("\n" + "=" * 60)
print("8. DUPLICATE ORDER IDs")
print("=" * 60)

duplicate_rows = df[
    df.duplicated(subset=["order_id"], keep=False)
]

print(duplicate_rows)


# Count duplicates that will be removed

duplicates_removed = df.duplicated(
    subset=["order_id"]
).sum()

print("\nNumber of duplicates removed:")
print(duplicates_removed)


# Remove duplicate Order IDs

df = df.drop_duplicates(
    subset=["order_id"],
    keep="first"
)

df = df.reset_index(drop=True)


# 9A. FILTER UNIT PRICE GREATER THAN 20,000

print("\n" + "=" * 60)
print("9A. ORDERS WHERE UNIT PRICE > 20,000")
print("=" * 60)

orders_over_20000 = df[
    df["unit_price"] > 20000
]

print(
    orders_over_20000[
        [
            "order_id",
            "customer_name",
            "category",
            "product",
            "unit_price",
            "status"
        ]
    ]
)


# 9B. UNIT PRICE > 10,000 AND STATUS = COMPLETED

print("\n" + "=" * 60)
print("9B. UNIT PRICE > 10,000 AND STATUS = COMPLETED")
print("=" * 60)

completed_orders = df.loc[
    (df["unit_price"] > 10000)
    & (df["status"] == "Completed"),
    [
        "customer_name",
        "category",
        "status"
    ]
]

print(completed_orders)


# 9C. CREATE TOTAL_AMOUNT COLUMN
#
# total_amount = quantity * unit_price - discount

df["total_amount"] = (
    df["quantity"] * df["unit_price"]
    - df["discount"]
)

print("\n" + "=" * 60)
print("9C. TOTAL AMOUNT")
print("=" * 60)

print(
    df[
        [
            "order_id",
            "quantity",
            "unit_price",
            "discount",
            "total_amount"
        ]
    ]
)


# 9D. CREATE CUSTOMER_TYPE COLUMN
# Bulk Buyer = quantity >= 3
# Regular Buyer = otherwise

df["customer_type"] = np.where(
    df["quantity"] >= 3,
    "Bulk Buyer",
    "Regular Buyer"
)

print("\n" + "=" * 60)
print("9D. CUSTOMER TYPE")
print("=" * 60)

print(
    df[
        [
            "customer_name",
            "quantity",
            "customer_type"
        ]
    ]
)


# 10. FINAL CLEAN DATASET

print("\n" + "=" * 60)
print("10. FINAL DATASET INFORMATION")
print("=" * 60)

print("Original shape:", original_shape)
print("Final shape:", df.shape)

print("\nFirst 10 rows of cleaned dataset:")
print(df.head(10))


# CHECK REMAINING MISSING VALUES

print("\n" + "=" * 60)
print("REMAINING MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())

# OPTIONAL: SAVE CLEANED DATASET

df.to_csv(
    "cleaned_pandas_dataset.csv",
    index=False
)

print("\nCleaned dataset saved as:")
print("cleaned_pandas_dataset.csv")