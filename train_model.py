import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

# Load and inspect the file
df = pd.read_csv("spam.csv", encoding="latin-1")

# Try to auto-detect correct columns
if 'v1' in df.columns and 'v2' in df.columns:
    df = df[['v1', 'v2']]
elif 'Category' in df.columns and 'Message' in df.columns:
    df = df[['Category', 'Message']]
    df = df.rename(columns={'Category': 'v1', 'Message': 'v2'})
else:
    raise Exception("❌ Could not find required columns in spam.csv")

# Label Encoding
df['label'] = df['v1'].map({'ham': 0, 'spam': 1})

# Features and Labels
X = df['v2']
y = df['label']

# Vectorization
cv = CountVectorizer()
X = cv.fit_transform(X)

# Model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the model and vectorizer
joblib.dump(model, 'spam_model.pkl')
joblib.dump(cv, 'vectorizer.pkl')

print("✅ Model and vectorizer saved successfully!")
