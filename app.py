from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model & vectorizer
clf = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        password = request.form["password"]
        X_input = vectorizer.transform([password])
        strength = clf.predict(X_input)[0]

        if strength == 0:
            prediction = "Weak"
        elif strength == 1:
            prediction = "Medium"
        else:
            prediction = "Strong"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

