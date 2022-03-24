import numpy as np
from logzero import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from ..base import ZooBase

np.random.seed(512)


class ZooRandomForest(ZooBase):
    def __init__(self):
        super().__init__()
        self._best_model = RandomForestClassifier()

    def fit(self, x, y):
        x, y = self._preprocess(x, y)

        param_grid = {
            'bootstrap': [True],
            'max_depth': [10, 50, 100, 250],
            'n_estimators': [25, 50, 100, 500]
        }
        # Create a based model
        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator=self._best_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(x, y)
        logger.info(f'best params: {grid_search.best_params_}')

        self._best_model = grid_search.best_estimator_

    def predict(self, x, training=False):
        if training:
            x, _ = self._preprocess(x)
        return self._best_model.predict(x)

    def predict_proba(self, x, training=False):
        if training:
            x, _ = self._preprocess(x)
        return self._best_model.predict_proba(x)

    def score(self, x, y, training=False):
        if training:
            x, y = self._preprocess(x, y)
        return self._best_model.score(x, y)

    def _preprocess(self, x, y=None):
        x_new = self._feature_extraction(x)

        y_new = None
        if y is not None:
            y_new = y.reset_index(drop=True)
            y_new = y_new.values.ravel()

        return x_new, y_new
