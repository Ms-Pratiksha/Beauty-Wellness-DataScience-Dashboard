import pandas as pd

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("Data/beauty_products_sample.csv")

# Load customer segmentation results
segments = pd.read_csv(
    "Data/beauty_products_customer_segments.csv"
)

print("========== SMART BEAUTY PRODUCT RECOMMENDATION SYSTEM ==========")


# ==========================================
# CUSTOMER INPUT
# ==========================================

skin_type = input(
    "Enter your skin type (Dry/Oily/Combination): "
).strip().title()

skin_concern = input(
    "Enter your skin concern "
    "(Acne/Dryness/Aging/Frizz/Pigmentation/Oiliness): "
).strip().title()

budget = float(input("Enter your budget: "))

preferred_category = input(
    "Enter preferred category (Skincare/Haircare/Makeup): "
).strip().title()


# ==========================================
# FIND SIMILAR CUSTOMER SEGMENT
# ==========================================

# Find customers with similar skin type and concern
similar_customers = segments[
    (segments["Skin_Type"].str.lower() == skin_type.lower())
    &
    (segments["Skin_Concern"].str.lower() == skin_concern.lower())
]

if len(similar_customers) > 0:

    customer_segment = similar_customers[
        "Customer_Segment"
    ].mode()[0]

else:

    customer_segment = None


# ==========================================
# FILTER BY CATEGORY
# ==========================================

category_products = df[
    df["Product_Category"].str.lower()
    == preferred_category.lower()
].copy()


# ==========================================
# FILTER BY BUDGET
# ==========================================

recommendations = category_products[
    category_products["Product_Price"] <= budget
].copy()


# ==========================================
# NO PRODUCT WITHIN BUDGET
# ==========================================

if recommendations.empty:

    print("\nNo products found within your selected budget.")

    if len(category_products) > 0:

        closest_product = category_products.loc[
            (category_products["Product_Price"] - budget)
            .abs()
            .idxmin()
        ]

        print("\nClosest available product:")
        print(
            f"Product ID: {closest_product['Product_ID']}"
        )
        print(
            f"Price: ₹{closest_product['Product_Price']}"
        )
        print(
            f"Rating: {closest_product['Rating']}"
        )

        difference = (
            closest_product["Product_Price"] - budget
        )

        if difference > 0:
            print(
                f"You need ₹{difference:.0f} more "
                "to purchase this product."
            )


# ==========================================
# RECOMMENDATION SCORE
# ==========================================

else:

    recommendations["Score"] = 0.0

    # Skin type match
    recommendations.loc[
        recommendations["Skin_Type"].str.lower()
        == skin_type.lower(),
        "Score"
    ] += 3

    # Skin concern match
    recommendations.loc[
        recommendations["Skin_Concern"].str.lower()
        == skin_concern.lower(),
        "Score"
    ] += 4

    # Rating score
    recommendations["Score"] += recommendations["Rating"]

    # Budget score
    recommendations["Budget_Score"] = (
        (budget - recommendations["Product_Price"])
        / budget
    ) * 2

    recommendations["Score"] += (
        recommendations["Budget_Score"]
    )


    # ==========================================
    # CUSTOMER SEGMENT INFLUENCE
    # ==========================================

    if customer_segment is not None:

        if customer_segment == 1:
            # Frequent budget customers
            recommendations.loc[
                recommendations["Product_Price"] <= budget * 0.75,
                "Score"
            ] += 1

        elif customer_segment == 2:
            # Premium customers
            recommendations.loc[
                recommendations["Rating"] >= 4.5,
                "Score"
            ] += 1

        else:
            # Regular customers
            recommendations.loc[
                recommendations["Product_Price"] <= budget,
                "Score"
            ] += 0.5


    # ==========================================
    # SORT RESULTS
    # ==========================================

    recommendations = recommendations.sort_values(
        by="Score",
        ascending=False
    )


    # ==========================================
    # DISPLAY RESULTS
    # ==========================================

    print("\n========== TOP RECOMMENDED PRODUCTS ==========")

    print(
        recommendations[
            [
                "Product_ID",
                "Product_Category",
                "Product_Price",
                "Rating",
                "Score"
            ]
        ].head(5).to_string(index=False)
    )


    print("\nRecommendation process completed!")