# USAGE
# Start the server:
# 	python run_front_server.py
# Submit a request via Python:
#	python simple_request.py

# import the necessary packages
import dill
import pandas as pd
import os
dill._dill._reverse_typemap['ClassType'] = type
#import cloudpickle
import flask
import logging
from logging.handlers import RotatingFileHandler
from time import strftime

# initialize our Flask application and the model
app = flask.Flask(__name__)
model = None

handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

def load_model(model_path):
	# load the pre-trained model
	global model
	with open(model_path, 'rb') as f:
		model = dill.load(f)
	print(model)

modelpath = "/app/app/models/logreg_pipeline.dill"
load_model(modelpath)

@app.route("/", methods=["GET"])
def general():
	return """Welcome to fraudelent prediction process. Please use 'http://<address>/predict' to POST"""

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}
	dt = strftime("[%Y-%b-%d %H:%M:%S]")
	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":

		age, anaemia, creatinine_phosphokinase = "", "", ""
		diabetes, ejection_fraction, high_blood_pressure = "", "", ""
		platelets, serum_creatinine, serum_sodium = "", "", ""
		sex, smoking, time = "", "", ""
		request_json = flask.request.get_json()
		if request_json["age"]:
			age = request_json['age']

		if request_json["anaemia"]:
			anaemia = request_json['anaemia']

		if request_json["creatinine_phosphokinase"]:
			creatinine_phosphokinase = request_json['creatinine_phosphokinase']
			
		if request_json["diabetes"]:
			diabetes = request_json['diabetes']

		if request_json["ejection_fraction"]:
			ejection_fraction = request_json['ejection_fraction']

		if request_json["high_blood_pressure"]:
			high_blood_pressure = request_json['high_blood_pressure']

		if request_json["platelets"]:
			platelets = request_json['platelets']

		if request_json["serum_creatinine"]:
			serum_creatinine = request_json['serum_creatinine']

		if request_json["serum_sodium"]:
			serum_sodium = request_json['serum_sodium']
			
		if request_json["sex"]:
			sex = request_json['sex']

		if request_json["smoking"]:
			smoking = request_json['smoking']

		if request_json["time"]:
			time = request_json['time']

		logger.info(f'{dt} Data: age={age}, anaemia={anaemia}, creatinine_phosphokinase={creatinine_phosphokinase}')
		logger.info(f'{dt} Data: diabetes={diabetes}, ejection_fraction={ejection_fraction}, high_blood_pressure={high_blood_pressure}')
		logger.info(f'{dt} Data: platelets={platelets}, serum_creatinine={serum_creatinine}, serum_sodium={serum_sodium}')
		logger.info(f'{dt} Data: sex={sex}, smoking={smoking}, time={time}')
		try:
			preds = model.predict_proba(pd.DataFrame({"age": [age],
												  "anaemia": [anaemia],
												  "creatinine_phosphokinase": [creatinine_phosphokinase],
												  "diabetes": [diabetes],
												  "ejection_fraction": [ejection_fraction],
												  "high_blood_pressure": [high_blood_pressure],
												  "platelets": [platelets],
												  "serum_creatinine": [serum_creatinine],
												  "serum_sodium": [serum_sodium],
												  "sex": [sex],
												  "smoking": [smoking],
												  "time": [time]}))
		except AttributeError as e:
			logger.warning(f'{dt} Exception: {str(e)}')
			data['predictions'] = str(e)
			data['success'] = False
			return flask.jsonify(data)

		data["predictions"] = preds[:, 1][0]
		# indicate that the request was a success
		data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading the model and Flask starting server..."
		"please wait until server has fully started"))
	port = int(os.environ.get('PORT', 8180))
	app.run(host='0.0.0.0', debug=True, port=port)
