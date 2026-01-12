import joblib

vectorizer = joblib.load("tfidf_vectorizer.pkl")
lr_model = joblib.load("lr_model.pkl")

def keyword_urgency_detection(text):
    high = ['urgent', 'asap', 'deadline', 'emergency', 'immediate', 'critical']
    medium = ['soon', 'important', 'priority', 'quick']
    text = text.lower()

    if any(word in text for word in high):
        return 'high'
    elif any(word in text for word in medium):
        return 'medium'
    return 'low'

def predict_urgency(text):
    text_vec = vectorizer.transform([text])
    ml_pred = lr_model.predict(text_vec)[0]

    keyword_pred = keyword_urgency_detection(text)
    if keyword_pred == 'high':
        return 'high'
    return ml_pred
