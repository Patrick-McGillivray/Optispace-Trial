

import numpy as np
from flask import Flask,render_template,request
import pickle


app = Flask(__name__)
model = pickle.load(open('Optispace_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict_attrition',methods=['POST'])
def predict_attrition():
    prediction=''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    result = model.predict(final_features)
    if int(result)==0:
        prediction='Yes'
    elif int(result)==1:
        prediction='No'
    else:
        prediction='Unknown'

    return render_template('index.html', prediction_text='Predicted attrition {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
