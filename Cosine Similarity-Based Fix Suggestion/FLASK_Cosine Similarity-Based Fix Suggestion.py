from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
df = pd.read_csv("balanced_bug_dataset.csv").dropna()

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    df['Buggy_Code'], df['Project'], test_size=0.2, random_state=42)

# Convert code into numerical features using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_tfidf, y_train)


# Function to suggest a fix using cosine similarity
def suggest_fix(buggy_code):
    try:
        tfidf_buggy = vectorizer.transform([buggy_code])
        tfidf_fixed = vectorizer.transform(df['Fixed_Code'])  # Match to Fixed Code instead of Buggy Code
        similarities = cosine_similarity(tfidf_buggy, tfidf_fixed).flatten()
        best_match_idx = similarities.argmax()
        return df.iloc[best_match_idx]['Fixed_Code']
    except Exception:
        return "No suitable fix found."


@app.route("/", methods=["GET", "POST"])
def index():
    suggested_fix = ""
    buggy_code = ""

    if request.method == "POST":
        buggy_code = request.form.get("buggy_code", "")
        if "reset" in request.form:
            buggy_code = ""
            suggested_fix = ""
        else:
            suggested_fix = suggest_fix(buggy_code)

    return render_template("index.html", buggy_code=buggy_code, suggested_fix=suggested_fix)


if __name__ == "__main__":
    app.run(debug=True)
