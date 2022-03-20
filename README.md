# ML Data Prep Zoo

* Create virtual envrionment and install dependencies

```python
conda create -n data-zoo-test python=3.7
conda activate data-zoo-test
pip install -r requirements.txt
```

* Old notebooks to run in sequence
    * notebooks/original_Random_Forest.ipynb
      * create model file as well as vectorizer file for Attribute Name
    * notebooks/original_featurizer.ipynb
      * feature engieering and type inference

* datazoo/ contains **reusable model and featurizer modules for Auto-Type-Inference** 
    * model/ contains the model file(s)
    * featurize/ contains the Featurizer
  
* New end-to-end notebook with modularized DataZoo modules
    * notebooks/new-end-to-end.ipynb
 