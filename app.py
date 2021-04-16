import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

trf1 = pickle.load(open('trf1.pkl', 'rb'))

trf2 = pickle.load(open('trf2.pkl', 'rb'))

trf3 = pickle.load(open('trf3.pkl', 'rb'))

scaler = pickle.load(open('scaler.pkl', 'rb'))

model = pickle.load(open('model.pkl', 'rb'))

def preprocess(lst=[]):
    lst = trf1.transform(lst)
    lst = trf2.transform(lst)
    lst = trf3.transform(lst)

    lst_flt = []
    for itm in lst[0]:
        lst_flt.append(float(itm))

    list_flt = [np.array(lst_flt)]
    list_flt = scaler.transform(list_flt)
    list_flt = np.append(arr=np.ones((np.shape(list_flt)[0], 1), dtype=int), values=list_flt, axis=1)
    return list_flt

@app.route('/')
def home():
    return render_template('form.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    int_features[0] = int_features[0].upper()
    int_features[5] = int_features[5].upper()
    int_features[6] = int_features[6].upper()
    final_features = [np.array(int_features)]
    final_features = preprocess(final_features)
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('result.html', prediction_text='Employee Monthly Salary should be Rs {}/-'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
