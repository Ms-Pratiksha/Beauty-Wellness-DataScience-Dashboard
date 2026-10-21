import base64
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Beauty & Wellness Data Science",
    page_icon="💄",
    layout="wide"
)

# =========================================================
# BACKGROUND IMAGE
# =========================================================

def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>

        /* Main website background */
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* Main dashboard content */
        .block-container {{
            background-color: rgba(255, 255, 255, 0.82);
            padding: 2rem;
            border-radius: 15px;
        }}

        /* Make headers clear */
        h1, h2, h3 {{
            font-weight: 700;
        }}

        /* Metric cards */
        [data-testid="stMetric"] {{
            background-color: rgba(255, 255, 255, 0.92);
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


# IMPORTANT:
# Keep background.png in the same folder as app.py
set_background("background.png")


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("Data/beauty_products_sample.csv")

sentiment_df = pd.read_csv(
    "Data/beauty_products_sentiment_results.csv"
)

segments_df = pd.read_csv(
    "Data/beauty_products_customer_segments.csv"
)


# =========================================================
# TITLE
# =========================================================

st.title("💄 Beauty & Wellness Data Science Dashboard")

st.write(
    "Data-driven customer analysis, sentiment analysis "
    "and personalized beauty product recommendations."
)

st.divider()


# =========================================================
# KEY METRICS
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Customers",
        len(df)
    )

with col2:
    st.metric(
        "Average Rating",
        f"{df['Rating'].mean():.2f} ⭐"
    )

with col3:
    positive_reviews = (
        df["Review_Sentiment"]
        .eq("Positive")
        .sum()
    )

    st.metric(
        "Positive Reviews",
        positive_reviews
    )

with col4:
    st.metric(
        "Product Categories",
        df["Product_Category"].nunique()
    )


st.divider()


# =========================================================
# PRODUCT CATEGORY ANALYSIS
# =========================================================

st.header("📊 Product Category Analysis")

col1, col2 = st.columns(2)

with col1:

    category_counts = (
        df["Product_Category"]
        .value_counts()
    )

    st.subheader("Products by Category")

    fig1, ax1 = plt.subplots(figsize=(7, 4))

    category_counts.plot(
        kind="bar",
        ax=ax1,
        color=["#FF69B4", "#9370DB", "#20B2AA"]
    )

    ax1.set_xlabel("Product Category")
    ax1.set_ylabel("Number of Products")
    ax1.set_title("Products by Category")

    plt.xticks(rotation=0)
    plt.tight_layout()

    st.pyplot(fig1)


with col2:

    category_rating = (
        df.groupby("Product_Category")["Rating"]
        .mean()
        .round(2)
    )

    st.subheader("Average Rating by Category")

    fig2, ax2 = plt.subplots(figsize=(7, 4))

    category_rating.plot(
        kind="bar",
        ax=ax2,
        color=["#FFA500", "#32CD32", "#6495ED"]
    )

    ax2.set_xlabel("Product Category")
    ax2.set_ylabel("Average Rating")
    ax2.set_title("Average Rating by Category")

    plt.xticks(rotation=0)
    plt.tight_layout()

    st.pyplot(fig2)


# =========================================================
# SENTIMENT ANALYSIS
# =========================================================

st.header("💬 Customer Review Sentiment")

sentiment_counts = (
    df["Review_Sentiment"]
    .value_counts()
)

fig3, ax3 = plt.subplots(figsize=(7, 4))

sentiment_counts.plot(
    kind="bar",
    ax=ax3,
    color=["#32CD32", "#FFD700", "#FF6347"]
)

ax3.set_xlabel("Sentiment")
ax3.set_ylabel("Number of Reviews")
ax3.set_title("Customer Review Sentiment")

plt.xticks(rotation=0)
plt.tight_layout()

st.pyplot(fig3)

st.write(
    "Positive reviews:",
    int(sentiment_counts.get("Positive", 0))
)

st.write(
    "Neutral reviews:",
    int(sentiment_counts.get("Neutral", 0))
)

st.write(
    "Negative reviews:",
    int(sentiment_counts.get("Negative", 0))
)


# =========================================================
# CUSTOMER SEGMENTATION
# =========================================================

st.header("👥 Customer Segmentation")

fig, ax = plt.subplots(figsize=(8, 5))

segment_colors = [
    "#4169E1",
    "#FF69B4",
    "#32CD32"
]

for i, segment in enumerate(
    sorted(segments_df["Customer_Segment"].unique())
):

    segment_data = segments_df[
        segments_df["Customer_Segment"] == segment
    ]

    ax.scatter(
        segment_data["Budget"],
        segment_data["Purchase_Frequency"],
        label=f"Segment {segment}",
        color=segment_colors[i],
        s=80
    )

ax.set_xlabel("Customer Budget")
ax.set_ylabel("Purchase Frequency")
ax.set_title("Customer Segmentation")
ax.legend()

plt.tight_layout()

st.pyplot(fig)


# =========================================================
# SECOND SEGMENT CHART
# =========================================================

st.subheader("📈 Customer Segment Distribution")

fig, ax = plt.subplots(figsize=(8, 5))

for segment in sorted(
    segments_df["Customer_Segment"].unique()
):

    segment_data = segments_df[
        segments_df["Customer_Segment"] == segment
    ]

    ax.scatter(
        segment_data["Budget"],
        segment_data["Purchase_Frequency"],
        label=f"Segment {segment}"
    )

ax.set_xlabel("Customer Budget")
ax.set_ylabel("Purchase Frequency")
ax.set_title("Customer Segmentation")

ax.legend()

plt.tight_layout()

st.pyplot(fig)


# =========================================================
# PRODUCT RECOMMENDATION
# =========================================================

st.header("🛍️ Personalized Product Recommendation")

st.write(
    "Enter customer preferences to find suitable products."
)

col1, col2 = st.columns(2)

with col1:

    skin_type = st.selectbox(
        "Skin Type",
        [
            "Dry",
            "Oily",
            "Combination"
        ]
    )

    skin_concern = st.selectbox(
        "Skin Concern",
        [
            "Acne",
            "Dryness",
            "Aging",
            "Frizz",
            "Pigmentation",
            "Oiliness"
        ]
    )


with col2:

    budget = st.number_input(
        "Budget (₹)",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100
    )

    preferred_category = st.selectbox(
        "Preferred Category",
        [
            "Skincare",
            "Haircare",
            "Makeup"
        ]
    )


# =========================================================
# RECOMMENDATION BUTTON
# =========================================================

if st.button(
    "🔍 Recommend Products",
    use_container_width=True
):

    recommendations = df[
        df["Product_Category"].str.lower()
        == preferred_category.lower()
    ].copy()

    # -----------------------------------------------------
    # Budget filtering
    # -----------------------------------------------------

    recommendations = recommendations[
        recommendations["Product_Price"] <= budget
    ].copy()


    # -----------------------------------------------------
    # No products available
    # -----------------------------------------------------

    if recommendations.empty:

        st.warning(
            "No products found within your selected budget."
        )

        category_products = df[
            df["Product_Category"].str.lower()
            == preferred_category.lower()
        ]

        if not category_products.empty:

            closest_product = category_products.loc[
                (
                    category_products["Product_Price"]
                    - budget
                ).abs().idxmin()
            ]

            st.info(
                f"Closest available product: "
                f"{closest_product['Product_ID']} "
                f"at ₹{closest_product['Product_Price']}"
            )


    else:

        # -------------------------------------------------
        # Create recommendation score
        # -------------------------------------------------

        recommendations["Score"] = 0.0


        # -------------------------------------------------
        # Skin type match
        # -------------------------------------------------

        recommendations.loc[
            recommendations["Skin_Type"].str.lower()
            == skin_type.lower(),
            "Score"
        ] += 3


        # -------------------------------------------------
        # Skin concern match
        # -------------------------------------------------

        recommendations.loc[
            recommendations["Skin_Concern"].str.lower()
            == skin_concern.lower(),
            "Score"
        ] += 4


        # -------------------------------------------------
        # Product rating
        # -------------------------------------------------

        recommendations["Score"] += (
            recommendations["Rating"]
        )


        # -------------------------------------------------
        # Budget suitability
        # -------------------------------------------------

        recommendations["Budget_Score"] = (
            (
                budget
                - recommendations["Product_Price"]
            )
            / budget
        ) * 2

        recommendations["Score"] += (
            recommendations["Budget_Score"]
        )


        # -------------------------------------------------
        # Sort recommendations
        # -------------------------------------------------

        recommendations = recommendations.sort_values(
            "Score",
            ascending=False
        )


        st.success(
            "Top recommended products found!"
        )


        # -------------------------------------------------
        # Display top 5 products
        # -------------------------------------------------

        result = recommendations[
            [
                "Product_ID",
                "Product_Category",
                "Product_Price",
                "Rating",
                "Score"
            ]
        ].head(5).copy()


        result.columns = [
            "Product ID",
            "Category",
            "Price (₹)",
            "Rating",
            "Recommendation Score"
        ]


        st.dataframe(
            result,
            use_container_width=True,
            hide_index=True
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Beauty & Wellness Data Science Project | "
    "Python • Pandas • Scikit-learn • Streamlit"
)