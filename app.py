import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("models/sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    review_text = ""

    if request.method == "POST":
        review_text = request.form["review"]
        vector = vectorizer.transform([review_text])
        prediction = model.predict(vector)[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"

    return render_template(
        "index.html",
        sentiment=sentiment,
        review_text=review_text
    )

if __name__ == "__main__":
    app.run(debug=True)