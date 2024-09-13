from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("electro100.pkl")

@app.route('/')
def index():
    return render_template('hii.html')

@app.route('/input')
def input_page():
    return render_template('two.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()

    num_fans = int(data.get('num_fans', 0)) 
    num_lights = int(data.get('num_lights', 0))
    num_tv = int(data.get('num_tv', 0))
    num_fridge = int(data.get('num_fridge', 0))
    num_ac = int(data.get('num_ac', 0))
    num_wm = int(data.get('num_wm', 0))
    hours_fans = int(data.get('hours_fans', 0))
    hours_lights = int(data.get('hours_lights', 0))
    hours_tv = int(data.get('hours_tv', 0))
    hours_fridge = int(data.get('hours_fridge', 0))
    hours_ac = int(data.get('hours_ac', 0))
    hours_wm = int(data.get('hours_wm', 0))

    input_features = [
        num_fans, num_lights, num_tv, num_fridge, num_ac,num_wm,
        hours_fans, hours_lights, hours_tv, hours_fridge, hours_ac,hours_wm
    ]

    prediction = model.predict([input_features])
    prediction_integer = "Approx   "+str(int(prediction[0]))+"  Rupees  "
    response_data = {
        'prediction': prediction_integer,
        'message': 'Prediction completed successfully!'
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)
