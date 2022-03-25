import os
import subprocess
import traceback
import joblib
from datetime import datetime

from flask import Flask, render_template, request
import pandas as pd
import shap
import logzero
from logzero import logger

logzero.logfile(os.path.join(os.path.dirname(__file__), "fb.log"))

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

explainer = shap.TreeExplainer(model._best_model)

fe_path = os.path.join(os.path.dirname(__file__), "artifacts", "featurizer.joblib")
fe = joblib.load(fe_path)

column_names = model.fe_columns
column_names.insert(0, "Column Name")
column_names.insert(1, "Predicted Column Type")
column_names.insert(2, "Prediction Probability")
column_names.insert(3, "Disguised Missing Value (DMV)")
column_names.append("attribute_name")


@app.route("/")
def index():
    return render_template("index.html", column_names=column_names, items=[])


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():

    items = []
    if request.method == 'POST':
        shap_flag = False
        if request.form['btnNavbarSearch'] == "Upload with SHAP Values":
            shap_flag = True

        file = request.files["file"]
        logger.info(f'Start uploading {file.filename}')
        df = pd.read_csv(file)

        attr_names = df.columns
        data_featurized = fe.featurize(df)
        y_rf = model.predict(data_featurized).tolist()
        y_rf_probs = model.predict_proba(data_featurized).tolist()

        for ind, attr in enumerate(attr_names):
            percent = round(y_rf_probs[ind][y_rf[ind]], 3)
            d = {'name': attr, 'type': dict_int_label[y_rf[ind]], 'prob': percent, 'dmv': ''}
            f_dict = data_featurized.iloc[ind][:25].to_dict()

            if shap_flag:
                all_shap_values = explainer.shap_values(data_featurized.iloc[ind], check_additivity=False)
                shap_values = all_shap_values[y_rf[ind]][:25]
                for i, key in enumerate(f_dict.keys()):
                    f_dict[key] = f'{f_dict[key]} (SHAP Value: {shap_values[i]})'

                f_dict['attribute_name'] = f'(SHAP Value: {sum(all_shap_values[y_rf[ind]][25:])})'

                logger.info(f'Done process SHAP value for {attr}')
            else:
                f_dict['attribute_name'] = ""

            d.update(f_dict)
            items.append(d)

        file_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        upload_path = f'upload/{file_name}'
        logger.info(f'Uploading file to {upload_path}')
        df.to_csv(upload_path, index=False)
        logger.info(f'Start processing DMV FOR {file_name}')
        dmv_df_list = []
        for dir_name, type_name in [('syntactic', '1'), ('numerical', '3')]:
            command = ['./FAHES', upload_path, f'dmv/{dir_name}', type_name]
            process = subprocess.Popen(command,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            try:
                stdout, stderr = process.communicate(timeout=30)
                logger.info(f'Done {dir_name}')
                logger.debug(f'stdout: {stdout}')
                logger.debug(f'stderr: {stderr}')
                dmv_df_list.append(pd.read_csv(f'dmv/{dir_name}/DMV_{file_name}'))
            except:
                logger.error(f'Error with {dir_name}: {traceback.format_exc()}')

        dmv_df = pd.concat(dmv_df_list)

        for d in items:
            attr_name = d['name']
            dmvs = dmv_df[dmv_df['Attribute Name'] == f'{attr_name}']
            if len(dmvs) > 0:
                for i in range(len(dmvs)):
                    if i == 0:
                        d['dmv'] += '['

                    d_v = dmvs.iloc[i]
                    d_type = 'Syntactic-Outlier' if d_v['Detecting Tool'] == 'SYN' else 'Numeric-Outlier'
                    d['dmv'] += f'(DMV: {d_v["DMV"]}; Type: {d_type}; Frequency: {d_v["Frequency"]})'

                    if i < len(dmvs) - 1:
                        d['dmv'] += ','

                    if i == len(dmvs) - 1:
                        d['dmv'] += ']'

        logger.info(f'Done processing DMV FOR {file_name}')

    return render_template("index.html", column_names=column_names, items=items)


if __name__ == '__main__':
   app.run(host="0.0.0.0", port=5001, debug=True)