from sklearn.feature_extraction.text import CountVectorizer
import indeed_scrape
import GrammarParser
import subprocess
import numpy as np
import logging
import os
from os import path
import numpy as np

data_dir = os.getenv('OPENSHIFT_DATA_DIR')
if data_dir is None:
    data_dir = os.getenv('PWD')

logging = logging.getLogger(__name__)
grammar = GrammarParser.GrammarParser()
ind = indeed_scrape.Indeed('kw')

class MissingKeywords(object):
    def __init__(self):
        self.stop_words = 'resume affirmative cover letter equal religion sex disibility veteran status sexual orientation and work ability http https www gender com org the'

    def pdf_to_text(self, infile):
        logging.debug("pdf_to_text, infile:%s" % infile)
        jar_file = os.path.join(data_dir, 'pdfbox-app-2.0.0-RC2.jar')
        cmd = "java -jar %(jar)s ExtractText -console %(infile)s"
        cmd = cmd % {'jar':jar_file,
                     'infile':infile
                     }

        process = subprocess.Popen(cmd, shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

        out, err = process.communicate()
        errcode = process.returncode

        if err or errcode:
            logging.error(err)
            raise Exception

        return out

    def make_row(self, kw, ind_cnt, res_cnt):
        row = '<tr><td>%s</td><td>%s</td><td>%s</td></tr>'
        row = row %(kw, ind_cnt, res_cnt)
        return row

    def vectorizer(self, corpus, max_fea, n_min, n_max):
        ind = indeed_scrape.Indeed(None)
        ind.stop_words = self.stop_words
        ind.add_stop_words()
        stop_words = ind.stop_words

        vectorizer = CountVectorizer(max_features=max_fea,
                                    max_df=0.80,
                                    min_df=5,
                                    lowercase=True,
                                    stop_words=stop_words,
                                    ngram_range=(n_min, n_max),
                                    analyzer='word',
                                    decode_error='ignore',
                                    strip_accents='unicode'
                                    )

        matrix = vectorizer.fit_transform(corpus)
        features = vectorizer.get_feature_names()

        return matrix, features, vectorizer

    def _trans_ind_agg_to_perc(self, mat):
        ind_mat = mat.toarray()
        # recall: mat is docs x features
        # we want count of features overall docs
        ind_cnt = ind_mat.T.sum(axis=1)
        # and really we want the percentage
        ind_perc = ind_cnt / float(ind_mat.shape[0])
        ind_perc = np.round(ind_perc, decimals=2)

        return ind_perc

    def main(self, resume_path, indeed_summaries):
        res_text = self.pdf_to_text(resume_path)

        bi_rows = self.bi_gram_analysis(res_text, indeed_summaries)
        uni_rows = self.unigram_analysis(res_text, indeed_summaries)

        return bi_rows, uni_rows

    def unigram_analysis(self, res_text, indeed_summaries):
        ind_mat, keywords, vec_obj = self.vectorizer(indeed_summaries, 10, 1, 1)
        ind_perc = self._trans_ind_agg_to_perc(ind_mat)

        res_mat = vec_obj.transform([res_text])
        # resume matrix is a 1 dim so no need to sum
        # or transpose
        res_mat = res_mat.toarray().squeeze()

        rows = self.make_rows(keywords, ind_perc, res_mat)
        return rows

    def bi_gram_analysis(self, res_text, indeed_summaries):
        ind_mat, keywords, vec_obj = self.vectorizer(indeed_summaries, 20, 2, 2)
        ind_perc = self._trans_ind_agg_to_perc(ind_mat)

        res_mat = vec_obj.transform([res_text])
        res_mat = res_mat.toarray().squeeze()

        rows = self.make_rows(keywords, ind_perc, res_mat)
        return rows

    def make_rows(self, keywords, ind_perc, res_mat):
        rows = ''
        for i in range(len(ind_perc)):
            rows += self.make_row(keywords[i], ind_perc[i], res_mat[i])

        return rows





