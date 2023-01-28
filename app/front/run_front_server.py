import json

from flask import Flask, render_template, redirect, url_for, request
from flask_wtf import FlaskForm
from requests.exceptions import ConnectionError
from wtforms import IntegerField, SelectField, StringField
from wtforms.validators import DataRequired

import urllib.request
import json

class ClientDataForm(FlaskForm):
    age = StringField('age', validators=[DataRequired()])
    anaemia = StringField('anaemia', validators=[DataRequired()])
    creatinine_phosphokinase = StringField('creatinine_phosphokinase', validators=[DataRequired()])
    diabetes = StringField('diabetes', validators=[DataRequired()])
    ejection_fraction = StringField('ejection_fraction', validators=[DataRequired()])
    high_blood_pressure = StringField('high_blood_pressure', validators=[DataRequired()])
    platelets = StringField('platelets', validators=[DataRequired()])
    serum_creatinine = StringField('serum_creatinine', validators=[DataRequired()])
    serum_sodium = StringField('serum_sodium', validators=[DataRequired()])
    sex = StringField('sex', validators=[DataRequired()])
    smoking = StringField('smoking', validators=[DataRequired()])
    time = StringField('time', validators=[DataRequired()])


app = Flask(__name__)
app.config.update(
    CSRF_ENABLED=True,
    SECRET_KEY='you-will-never-guess',
)

def get_prediction(age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time):
    body = {'age': age,
                            'anaemia': anaemia,
                            'creatinine_phosphokinase': creatinine_phosphokinase,
                            'diabetes': diabetes,
                            'ejection_fraction': ejection_fraction,
                            'high_blood_pressure': high_blood_pressure,
                            'platelets': platelets,
                            'serum_creatinine': serum_creatinine,
                            'serum_sodium': serum_sodium,
                            'sex': sex,
                            'smoking': smoking,
                            'time': time}

    myurl = "http://0.0.0.0:8180/predict"
    req = urllib.request.Request(myurl)
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    jsondata = json.dumps(body)
    jsondataasbytes = jsondata.encode('utf-8')   # needs to be bytes
    req.add_header('Content-Length', len(jsondataasbytes))
    #print (jsondataasbytes)
    response = urllib.request.urlopen(req, jsondataasbytes)
    return json.loads(response.read())['predictions']

@app.route("/")
def index():
    return render_template('index.html')


@app.route('/predicted/<response>')
def predicted(response):
    response = json.loads(response)
    print(response)
    return render_template('predicted.html', response=response)


@app.route('/predict_form', methods=['GET', 'POST'])
def predict_form():
    form = ClientDataForm()
    data = dict()
    if request.method == 'POST':
        data['age'] = request.form.get('age')
        data['anaemia'] = request.form.get('anaemia')
        data['creatinine_phosphokinase'] = request.form.get('creatinine_phosphokinase')
        data['diabetes'] = request.form.get('diabetes')
        data['ejection_fraction'] = request.form.get('ejection_fraction')
        data['high_blood_pressure'] = request.form.get('high_blood_pressure')
        data['platelets'] = request.form.get('platelets')
        data['serum_creatinine'] = request.form.get('serum_creatinine')
        data['serum_sodium'] = request.form.get('serum_sodium')
        data['sex'] = request.form.get('sex')
        data['smoking'] = request.form.get('smoking')
        data['time'] = request.form.get('time')


        try:
            response = str(get_prediction(data['age'],
                                      data['anaemia'],
                                      data['creatinine_phosphokinase'],
                                      data['diabetes'],
                                      data['ejection_fraction'],
                                      data['high_blood_pressure'],
                                      data['platelets'],
                                      data['serum_creatinine'],
                                      data['serum_sodium'],
                                      data['sex'],
                                      data['smoking'],
                                      data['time']))
            print(response)
        except ConnectionError:
            response = json.dumps({"error": "ConnectionError"})
        return redirect(url_for('predicted', response=response))
    return render_template('form.html', form=form)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8181, debug=True)
