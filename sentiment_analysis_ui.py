import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import gradio as gr

# Load dataset
data = pd.read_csv("customer_reviews.csv")
X = data['review']
y = data['sentiment']

# Vectorization
tfidf = TfidfVectorizer(stop_words='english')
X_vec = tfidf.fit_transform(X)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Function for prediction
def predict_sentiment(text):
    vec = tfidf.transform([text])
    pred = model.predict(vec)[0]
    return f"Predicted Sentiment: {pred.capitalize()}"

# Improved Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown(
        """
        # 🌟 Customer Review Sentiment Analyzer
        Enter a review below to analyze the sentiment using machine learning.
        The result will indicate whether the sentiment is **Positive**, **Neutral**, or **Negative**.
        """
    )
    with gr.Row():
        review_input = gr.Textbox(lines=4, placeholder="Type your review here...", label="✍️ Customer Review")
    with gr.Row():
        output = gr.Textbox(label="📊 Predicted Sentiment")
    with gr.Row():
        analyze_btn = gr.Button("🔍 Analyze Sentiment")

    analyze_btn.click(fn=predict_sentiment, inputs=review_input, outputs=output)

interface.launch()
