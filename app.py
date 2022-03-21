import os
import joblib

from flask import Flask, render_template, request
import pandas as pd


app = Flask(__name__,
            static_url_path='',
            static_folder='web/static',
            template_folder='web/templates')

dict_int_label = {
    0: 'numeric',
    1: 'categorical',
    2: 'datetime',
    3: 'sentence',
    4: 'url',
    5: 'embedded-number',
    6: 'list',
    7: 'not-generalizable',
    8: 'context-specific'
}

model_path = os.path.join(os.path.dirname(__file__), "artifacts", "model_rf.joblib")
model = joblib.load(model_path)

fe_path = os.path.join(os.path.dirname(__file__), "artifacts", "featurizer.joblib")
fe = joblib.load(fe_path)

column_names = model.fe_columns
column_names.insert(0, "Column Name")
column_names.insert(1, "Predicted Column Type")


@app.route("/")
def index():
    return render_template("index.html", column_names=column_names, items=[])


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    items = []
    if request.method == 'POST':
        file = request.files["file"]
        df = pd.read_csv(file)

        attr_names = df.columns
        data_featurized = fe.featurize(df)
        y_rf = model.predict(data_featurized).tolist()

        for ind, attr in enumerate(attr_names):
            d = {'name': attr, 'type': dict_int_label[y_rf[ind]]}
            f_dict = data_featurized.iloc[ind][:len(column_names)-2].to_dict()
            d.update(f_dict)
            items.append(d)

    return render_template("index.html", column_names=column_names, items=items)

if __name__ == '__main__':
   app.run(host="0.0.0.0", port=5001, debug=False)