import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# 1. Load the dataset
# -------------------------------------------------

file_path = "Data/beauty_products_sample.csv"

df = pd.read_csv(file_path)

print("\n========== BEAUTY & WELLNESS DATASET ==========")

# Display first 5 records
print("\nFirst 5 records:")
print(df.head())

# Dataset dimensions
print("\nDataset shape:")
print(df.shape)

# Column names
print("\nColumns:")
print(df.columns.tolist())

# -------------------------------------------------
# 2. Basic information
# -------------------------------------------------

print("\n========== DATA INFORMATION ==========")

print(df.info())

# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# Check duplicate records
print("\nDuplicate records:")
print(df.duplicated().sum())

# -------------------------------------------------
# 3. Basic statistics
# -------------------------------------------------

print("\n========== STATISTICAL SUMMARY ==========")

print(df.describe())

# -------------------------------------------------
# 4. Product category analysis
# -------------------------------------------------

print("\n========== PRODUCT CATEGORY ANALYSIS ==========")

category_counts = df["Product_Category"].value_counts()

print(category_counts)

# -------------------------------------------------
# 5. Skin type analysis
# -------------------------------------------------

print("\n========== SKIN TYPE ANALYSIS ==========")

skin_counts = df["Skin_Type"].value_counts()

print(skin_counts)

# -------------------------------------------------
# 6. Sentiment analysis summary
# -------------------------------------------------

print("\n========== REVIEW SENTIMENT ==========")

sentiment_counts = df["Review_Sentiment"].value_counts()

print(sentiment_counts)

# -------------------------------------------------
# 7. Average rating by category
# -------------------------------------------------

print("\n========== AVERAGE RATING ==========")

average_rating = df.groupby(
    "Product_Category"
)["Rating"].mean()

print(average_rating)

# -------------------------------------------------
# 8. Average price by category
# -------------------------------------------------

print("\n========== AVERAGE PRICE ==========")

average_price = df.groupby(
    "Product_Category"
)["Product_Price"].mean()

print(average_price)

# -------------------------------------------------
# 9. Visualization 1
# Product Category Distribution
# -------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Product_Category"
)

plt.title("Distribution of Beauty Product Categories")
plt.xlabel("Product Category")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "images/product_category_distribution.png"
)

plt.show()

# -------------------------------------------------
# 10. Visualization 2
# Skin Type Distribution
# -------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Skin_Type"
)

plt.title("Customer Distribution by Skin Type")
plt.xlabel("Skin Type")
plt.ylabel("Number of Customers")

plt.tight_layout()

plt.savefig(
    "images/skin_type_distribution.png"
)

plt.show()

# -------------------------------------------------
# 11. Visualization 3
# Review Sentiment
# -------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="Review_Sentiment"
)

plt.title("Customer Review Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.savefig(
    "images/review_sentiment_distribution.png"
)

plt.show()

# -------------------------------------------------
# 12. Visualization 4
# Rating by Product Category
# -------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Product_Category",
    y="Rating"
)

plt.title("Product Ratings by Category")
plt.xlabel("Product Category")
plt.ylabel("Rating")

plt.tight_layout()

plt.savefig(
    "images/rating_by_category.png"
)

plt.show()

print("\n========== ANALYSIS COMPLETED ==========")
print("Charts have been saved inside the images folder.")