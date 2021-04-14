import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
trf1 = pickle.load(open('trf1.pkl', 'rb'))
trf2 = pickle.load(open('trf2.pkl', 'rb'))

def preprocess(lst=[]):
    lst[0][0] = lst[0][0].upper()
    lst[0][5] = lst[0][5].upper()
    lst = trf1.transform(lst)
    lst = trf2.transform(lst)
    print(lst)

    lst_flt = []
    for itm in lst[0]:
        lst_flt.append(float(itm))
    print(lst_flt)

    list_flt = [np.array(lst_flt)]
    list_flt = np.append(arr=np.ones((np.shape(list_flt)[0], 1), dtype=int), values=list_flt, axis=1)
    print(list_flt)
    return list_flt

@app.route('/')
def home():
    return render_template('form2.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    For rendering results on HTML GUI
    """
    int_features = [x for x in request.form.values()]
    int_features[5] = int_features[5].upper()
    final_features = [np.array(int_features)]
    final_features = preprocess(final_features)
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('result.html', prediction_text='Employee Monthly Salary should be Rs {}/-'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
