import pandas as pd

# Load dataset
df = pd.read_csv("Data/beauty_products_sample.csv")

# Simple sentiment word lists
positive_words = [
    "good",
    "excellent",
    "amazing",
    "effective",
    "beautiful",
    "soft",
    "healthy",
    "great",
    "satisfied",
    "smooth",
    "brighter",
    "quality"
]

negative_words = [
    "bad",
    "disappointing",
    "disappointment",
    "not",
    "did not",
    "unhappy",
    "average"
]


def predict_sentiment(review):

    review = review.lower()

    positive_score = 0
    negative_score = 0

    for word in positive_words:
        if word in review:
            positive_score += 1

    for word in negative_words:
        if word in review:
            negative_score += 1

    if positive_score > negative_score:
        return "Positive"

    elif negative_score > positive_score:
        return "Negative"

    else:
        return "Neutral"


# Apply sentiment analysis
df["Predicted_Sentiment"] = df["Review_Text"].apply(
    predict_sentiment
)

# Display results
print("\n========== SENTIMENT ANALYSIS ==========")

print(
    df[
        [
            "Product_ID",
            "Review_Text",
            "Review_Sentiment",
            "Predicted_Sentiment"
        ]
    ]
)

# Compare original and predicted sentiment
print("\n========== SENTIMENT COMPARISON ==========")

comparison = pd.crosstab(
    df["Review_Sentiment"],
    df["Predicted_Sentiment"]
)

print(comparison)

# Save results
df.to_csv(
    "Data/beauty_products_sentiment_results.csv",
    index=False
)

print("\nSentiment analysis completed!")
print("Results saved to Data/beauty_products_sentiment_results.csv")