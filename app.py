from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('spam_model.pkl')  # ✅ Make sure this file exists in the same folder

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        message = request.form.get('message')  # Get input from form
        output = model.predict([message])      # Predict using model

        # Format output
        if output == [0]:
            result = "This Message is Not a SPAM Message."
        else:
            result = "This Message is a SPAM Message."

        return render_template('index.html', result=result, message=message)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
