import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from logzero import logger


class ZooBase(object):
    def __init__(self):
        self.vectorizer = CountVectorizer(ngram_range=(2, 2), analyzer='char')

    def _process_stats(self, data):
        data1 = data[['total_vals', 'num_nans', '%_nans', 'num_of_dist_val', '%_dist_val', 'mean', 'std_dev', 'min_val',
                      'max_val', 'has_delimiters', 'has_url', 'has_email', 'has_date', 'mean_word_count',
                      'stdev_word_count', 'mean_stopword_total', 'stdev_stopword_total',
                      'mean_char_count', 'stdev_char_count', 'mean_whitespace_count',
                      'stdev_whitespace_count', 'mean_delim_count', 'stdev_delim_count',
                      'is_list', 'is_long_sentence']]
        data1 = data1.reset_index(drop=True)
        data1 = data1.fillna(0)

        return data1

    def _feature_extraction(self, data, data1):
        arr = data['Attribute_name'].values
        arr = [str(x) for x in arr]

        if self._is_vectorizer_fitted():
            logger.info('vectorizer already fitted. Doing transform')
            X = self.vectorizer.transform(arr)
        else:
            logger.info('vectorizer not fitted. Doing fit and transform')
            X = self.vectorizer.fit_transform(arr)

        attr_df = pd.DataFrame(X.toarray())

        data2 = pd.concat([data1, attr_df], axis=1, sort=False)
        data2 = data2.fillna(0)

        logger.info(f'total length of from feature extraction: {len(data2)}')

        return data2

    def _is_vectorizer_fitted(self):
        try:
            self.vectorizer.get_feature_names_out()
        except:
            logger.warn('vectorizer not fitted yet')
            return False

        return True
