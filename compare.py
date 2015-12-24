import indeed_scrape
import GrammarParser
import subprocess
import numpy as np
import logging

logging = logging.getLogger(__name__)
grammar = GrammarParser.GrammarParser()
ind = indeed_scrape.Indeed('kw')

class MissingKeywords(object):
    def __init__(self):
        pass

    def pdf_to_text(self, infile):
        cmd = "java -jar pdfbox-app-2.0.0-RC2.jar ExtractText -console %s"
        cmd = cmd % (infile)

        process = subprocess.Popen(cmd, shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

        out, err = process.communicate()
        errcode = process.returncode

        if err or errcode:
            logging.error(err)
            raise Exception

        return out

    def make_rows(self, words):
        row = u'<tr><td>%s</td></tr>'
        rows = u''

        for w in words:
            row = row % w
            rows.append(row)

        return rows

    def main(self, resume_path, indeed_kws):

        text = self.pdf_to_text(resume_path)
        resume_kw = grammar.main(text)
        resume_kw = resume_kw.split(' ')

        _, kw = ind.vectorizer(indeed_kws, n_min=1, n_max=1, max_features=100,
                max_df=0.65, min_df=0.01)

        intersect = np.intersect1d(kw, resume_kw)

        for word in intersect:
            resume_kw.remove(word)

        return make_rows(resume_kw)
