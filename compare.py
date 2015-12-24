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
        row = '<tr><td>%s</td></tr>'
        rows = ''

        for w in words:
            rows += row % w

        return rows

    def main(self, resume_path, indeed_kws):

        text = self.pdf_to_text(resume_path)
        resume_kw = grammar.main(text)
        resume_kw = resume_kw.split(' ')

        indeed_kws = indeed_scrape.Indeed.len_tester(indeed_kws, thres=4)
        temp = []
        for row in indeed_kws:
            temp.append(grammar.main(row))

        _, kw = ind.vectorizer(temp, n_min=1, n_max=1, max_features=60,
                max_df=0.65, min_df=0.01)

        intersect = np.intersect1d(resume_kw, kw)

        for word in intersect:
            kw.remove(word)

        rows = self.make_rows(kw)

        return rows
