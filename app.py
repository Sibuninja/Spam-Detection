from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('spam_model.pkl')
cv = joblib.load('vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        message = request.form.get('message')
        if message:
            data = cv.transform([message])
            output = model.predict(data)
            result = "This Message is a SPAM Message." if output[0] == 1 else "This Message is Not a SPAM Message."
            return render_template('index.html', result=result, message=message)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
