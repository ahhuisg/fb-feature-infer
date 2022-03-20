import re
import random

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from ..base import ZooBase


del_pattern = r'([^,;\|]+[,;\|]{1}[^,;\|]+){1,}'
del_reg = re.compile(del_pattern)

delimeters = r"(,|;|\|)"
delimeters = re.compile(delimeters)

url_pat = r"(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?"
url_reg = re.compile(url_pat)

email_pat = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,6}\b"
email_reg = re.compile(email_pat)

stop_words = set(stopwords.words('english'))


class ZooFeaturizer(ZooBase):
    def __init__(self, vectorizer):
        super().__init__()
        self.vectorizer = vectorizer

    def featurize(self, df):
        featured_df = self._featurize_df(df)
        featured_df2 = self._process_stats(featured_df)
        return self._feature_extraction(featured_df, featured_df2)

    def _summary_stats(self, dat, key_s):
        for col in key_s:
            nans = np.count_nonzero(pd.isnull(dat[col])) # number of nan value
            dist_val = len(pd.unique(dat[col].dropna())) # number of unqiue none-nan value
            total_val = len(dat[col]) # total number of value

            mean = 0
            std_dev = 0
            min_val = 0
            max_val = 0
            if is_numeric_dtype(dat[col]):
                mean = np.mean(dat[col])
                if not pd.isnull(mean):
                    std_dev = np.std(dat[col])
                    min_val = float(np.min(dat[col]))
                    max_val = float(np.max(dat[col]))
                else:
                    mean = 0

            distinct_val_ratio = dist_val / total_val * 100.0
            nan_val_ratio = dist_val / total_val * 100.0

            yield [total_val, nans, dist_val, mean, std_dev, min_val, max_val, distinct_val_ratio, nan_val_ratio]

    def _get_sample(self, dat, key_s):
        for name in key_s:
            uniq_vals = pd.unique(dat[name])
            yield random.choices(uniq_vals, k=5)

    def _featurize_df(self, df):

        stats = []
        attribute_name = []
        sample = []

        keys = list(df.keys())

        attribute_name.extend(keys)
        summary_stat_result = self._summary_stats(df, keys)
        stats.extend(summary_stat_result)

        samples = self._get_sample(df, keys)
        sample.extend(samples)

        csv_names = ['Attribute_name', 'total_vals', 'num_nans', 'num_of_dist_val', 'mean', 'std_dev', 'min_val',
                     'max_val', '%_dist_val', '%_nans', 'sample_1', 'sample_2', 'sample_3', 'sample_4', 'sample_5'
                     ]
        curdf = pd.DataFrame(columns=csv_names)

        for i in range(len(attribute_name)):
            val_append = []
            val_append.append(attribute_name[i])
            val_append.extend(stats[i])

            val_append.extend(sample[i])

            curdf.loc[i] = val_append

        for row in curdf.itertuples():

            curlst = [row[11], row[12], row[13], row[14], row[15]]

            delim_cnt, url_cnt, email_cnt, date_cnt = 0, 0, 0, 0
            chars_totals, word_totals, stopwords, whitespaces, delims_count = [], [], [], [], []

            for value in curlst:
                if del_reg.match(str(value)):  delim_cnt += 1
                if url_reg.match(str(value)):  url_cnt += 1
                if email_reg.match(str(value)):  email_cnt += 1

                try:
                    _ = pd.Timestamp(value)
                    date_cnt += 1
                except ValueError:
                    date_cnt += 0

                word_totals.append(len(str(value).split(' ')))
                chars_totals.append(len(str(value)))
                whitespaces.append(str(value).count(' '))
                delims_count.append(len(delimeters.findall(str(value))))

                tokenized = word_tokenize(str(value))
                stopwords.append(len([w for w in tokenized if w in stop_words]))

            index = row[0]
            bool_fields = [
                ('has_delimiters', delim_cnt),
                ('has_url', url_cnt),
                ('has_email', email_cnt),
                ('has_date', date_cnt)
            ]

            for field, count in bool_fields:
                curdf.at[index, field] = True if count > 2 else False

            mean_std_fields = [
                ('word_count', word_totals),
                ('stopword_total', stopwords),
                ('char_count', chars_totals),
                ('whitespace_count', whitespaces),
                ('delim_count', delims_count)
            ]
            for field, lst in mean_std_fields:
                curdf.at[index, f'mean_{field}'] = np.mean(lst)
                curdf.at[index, f'stdev_{field}'] = np.std(lst)

            if curdf.at[index, 'has_delimiters'] and curdf.at[index, 'mean_char_count'] < 100:
                curdf.at[index, 'is_list'] = True
            else:
                curdf.at[index, 'is_list'] = False

            if curdf.at[index, 'mean_word_count'] > 10:
                curdf.at[index, 'is_long_sentence'] = True
            else:
                curdf.at[index, 'is_long_sentence'] = False

        return curdf
