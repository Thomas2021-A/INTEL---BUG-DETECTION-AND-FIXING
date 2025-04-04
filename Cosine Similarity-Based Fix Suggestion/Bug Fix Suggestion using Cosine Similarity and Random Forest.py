import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("balanced_bug_dataset.csv")

# Fill any missing values
df = df.dropna()

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

# Evaluate model performance
y_pred = clf.predict(X_test_tfidf)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# Function to suggest a fix using cosine similarity
def suggest_fix(buggy_code, df, vectorizer):
    tfidf_buggy = vectorizer.transform([buggy_code])
    tfidf_fixed = vectorizer.transform(df['Fixed_Code'])  # Match to Fixed Code instead of Buggy Code
    similarities = cosine_similarity(tfidf_buggy, tfidf_fixed).flatten()
    best_match_idx = similarities.argmax()
    return df.iloc[best_match_idx]['Fixed_Code']


# Continuous input loop with exit condition
print("\nEnter 'exit' or 'quit' to stop the program.")
while True:
    # Get user input
    user_buggy_code = input("\nEnter your buggy code: ").strip()

    # Check for exit keywords
    if user_buggy_code.lower() in ["exit", "quit"]:
        print("Exiting the program. Goodbye!")
        break

    # Suggest a fix for the entered buggy code
    print("\nBuggy Code:\n", user_buggy_code)
    try:
        suggested_fix = suggest_fix(user_buggy_code, df, vectorizer)
        print("Suggested Fix:\n", suggested_fix)
    except Exception as e:
        print("An error occurred while suggesting a fix:", str(e))