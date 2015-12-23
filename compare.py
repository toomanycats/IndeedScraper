import indeed_scrape
import GrammarParser
import subprocess
import numpy as np

grammar = GrammarParser.GrammarParser()
ind = indeed_scrape.Indeed('kw')

class MissingKeywords(object):
    def __init__(self):
        pass

    def run_pdf_to_text(self, infile):
        cmd = "java -jar pdfbox-app-2.0.0-RC2.jar ExtractText -console %s"
        cmd = cmd % (infile)

        process = subprocess.Popen(cmd, shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

        out, err = process.communicate()
        errcode = process.returncode

        if err or errcode:
            print err, errcode
            raise Exception

        return out

    def make_table(self, intersection):
        row = u'<tr><td>%s</td><td>%s</td></tr>'
        rows = u''

        for kw in intersection:
            rows += row % (kw)

        return rows

    def main(self, docs):
        infile = "/home/daniel/git_resume/d.cuneo.2015.pdf"

        text = run_pdf_to_text(infile)
        resume_kw = grammar.main(text)
        resume_kw = resume_kw.split(' ')

        # this monogram analysis is not performed in web_version.py
        # only the stem version when clicked
        #TODO: perform this analys in web_version and save to pickle

        _, kw = ind.vectorizer(docs, n_min=1, n_max=1, max_features=100,
                max_df=compute_max_df(sess_dict['type_'], sess_dict['num_urls']))

        intersect = np.intersect1d(kw, resume_kw)

        for word in intersect:
            resume_kw.remove(word)

        return resume_kw
