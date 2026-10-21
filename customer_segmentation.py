import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("Data/beauty_products_sample.csv")

print("========== CUSTOMER SEGMENTATION ==========")

# Select customer-related numerical variables
features = [
    "Age",
    "Budget",
    "Purchase_Frequency"
]

X = df[features]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create K-Means model
kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

# Predict customer clusters
df["Customer_Segment"] = kmeans.fit_predict(X_scaled)

# Display customers and their segments
print("\nCustomer Segments:")
print(
    df[
        [
            "Customer_ID",
            "Age",
            "Budget",
            "Purchase_Frequency",
            "Preferred_Category",
            "Customer_Segment"
        ]
    ].to_string(index=False)
)

# Segment summary
print("\n========== SEGMENT SUMMARY ==========")

segment_summary = df.groupby("Customer_Segment")[
    ["Age", "Budget", "Purchase_Frequency"]
].mean()

print(segment_summary)

# Save results
df.to_csv(
    "Data/beauty_products_customer_segments.csv",
    index=False
)

# Create visualization
plt.figure(figsize=(8, 6))

plt.scatter(
    df["Budget"],
    df["Purchase_Frequency"],
    c=df["Customer_Segment"]
)

plt.xlabel("Customer Budget")
plt.ylabel("Purchase Frequency")
plt.title("Customer Segmentation")
plt.savefig(
    "images/customer_segmentation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nCustomer segmentation completed!")
print(
    "Results saved to "
    "Data/beauty_products_customer_segments.csv"
)
print(
    "Chart saved to "
    "images/customer_segmentation.png"
)
