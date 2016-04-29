import indeed_scrape
import GrammarParser
import subprocess
import numpy as np
import logging
import os
from os import path

data_dir = os.getenv('OPENSHIFT_DATA_DIR')
if data_dir is None:
    data_dir = os.getenv('PWD')

logging = logging.getLogger(__name__)
grammar = GrammarParser.GrammarParser()
ind = indeed_scrape.Indeed('kw')

class MissingKeywords(object):
    def __init__(self):
        pass

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

    def make_rows(self, words):
        row = '<tr><td>%s</td></tr>'
        rows = ''

        for w in words:
            rows += row % w

        return rows

    def decode(self, text):
        try:
            text = text.encode("ascii", "ignore")
        except:
            text = text.decode("utf-8", "ignore").encode("ascii", "ignore")
        finally:
            return text

    def _len_tester(self, row):
            row = row.split(" ")
            row = indeed_scrape.Indeed.len_tester(row, thres=4)
            row = " ".join(row)

            return row

    def main(self, resume_path, indeed_summaries):
        text = self.pdf_to_text(resume_path)
        resume_kw = grammar.main(text)
        resume_kw = resume_kw.split(' ')

        summaries = map(grammar.main, indeed_summaries)
        summaries = map(self._len_tester, summaries)

        _, job_kw = ind.vectorizer(summaries, n_min=1, n_max=1, max_features=60,
                max_df=0.65, min_df=0.01)

        intersect = np.intersect1d(resume_kw, job_kw)

        for word in intersect:
            job_kw.remove(word)

        rows = self.make_rows(job_kw)

        return rows
