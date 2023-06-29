from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the trained machine learning model
model = pickle.load(open("project.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    date_time = request.form.get("DATE_TIME")
    cb_flow = float(request.form.get("CB_FLOW"))
    cb_press = float(request.form.get("CB_PRESS"))
    cb_temp = float(request.form.get("CB_TEMP"))
    steam_flow = float(request.form.get("STEAM_FLOW"))
    steam_temp = float(request.form.get("STEAM_TEMP"))
    steam_press = float(request.form.get("STEAM_PRESS"))
    o2_press = float(request.form.get("O2_PRESS"))
    o2_flow = float(request.form.get("O2_FLOW"))
    o2_per = float(request.form.get("O2_PER"))
    pci = float(request.form.get("PCI"))
    atm_humid = float(request.form.get("ATM_HUMID"))
    hb_temp = float(request.form.get("HB_TEMP"))
    hb_press = float(request.form.get("HB_PRESS"))
    top_press = float(request.form.get("TOP_PRESS"))
    top_temp1 = float(request.form.get("TOP_TEMP1"))
    top_spray = float(request.form.get("TOP_SPRAY"))
    top_temp = float(request.form.get("TOP_TEMP"))
    top_press_1 = float(request.form.get("TOP_PRESS_1"))
    h2 = float(request.form.get("H2"))
    skin_temp_avg = float(request.form.get("SKIN_TEMP_AVG"))
    co = float(request.form.get("CO"))
    co2 = float(request.form.get("CO2"))

    # Create a numpy array of the input values
    input_data = np.array([
        cb_flow, cb_press, cb_temp, steam_flow, steam_temp, steam_press,
        o2_press, o2_flow, o2_per, pci, atm_humid, hb_temp, hb_press,
        top_press, top_temp1, top_spray, top_temp, top_press_1, h2,
        skin_temp_avg, co, co2
    ]).reshape(1, -1)

    # Make the prediction using the model
    prediction = model.predict(input_data)

    # Prepare the prediction results
    prediction_text = f"Prediction: {prediction}"

    # Calculate CO:CO2 ratios for different hours
    hours = range(1, 5)
    co_ratios = [co / (co2 * h) for h in hours]

    # Create a bar plot to visualize the ratios
    plt.bar(hours, co_ratios)
    plt.xlabel('Hours')
    plt.ylabel('CO:CO2 Ratio')
    plt.title('CO:CO2 Ratio After Different Hours')
    plt.xticks(hours)
    
    # Save the plot to a BytesIO object
    img_bytes = io.BytesIO()
    plt.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    
    # Encode the image bytes as base64
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    
    # Calculate the ratios with labels for display
    ratio_labels = [f"After {h} hour(s), CO:CO2 ratio is {ratio:.2f}" for h, ratio in zip(hours, co_ratios)]
    
    return redirect(url_for('result', prediction_text=prediction_text, img_base64=img_base64, ratio_labels=ratio_labels))

@app.route('/result')
def result():
    prediction_text = request.args.get('prediction_text')
    img_base64 = request.args.get('img_base64')
    ratio_labels = request.args.getlist('ratio_labels')

    # Render the result page with the prediction result, image, and ratios
    return render_template('result.html', prediction_text=prediction_text, img_base64=img_base64, ratio_labels=ratio_labels)

if __name__ == '__main__':
    app.run(debug=True)