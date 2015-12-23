import indeed_scrape
import GrammarParser
import subprocess
import web_version
import numpy as np

grammar = GrammarParser.GrammarParser()
ind = indeed_scrape.Indeed('kw')

def run_pdf_to_text(infile, outfile):
    cmd = "java -jar pdfbox-app-2.0.0-RC2.jar ExtractText %s %s"
    cmd = cmd % (infile, outfile)

    process = subprocess.Popen(cmd, shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)

    out, err = process.communicate()
    errcode = process.returncode

    if err or errcode:
        print err, errcode
        raise Exception

def main():
    infile = "test.pdf"
    text_file = "test.txt"
    sess_dict = web_version.get_sess()


    run_pdf_to_text(infile, text_file)
    with open(text_file) as f:
        text = f.read()

    resume_kw = grammar.main(text)
    resume_kw = resume_kw.split(' ')

    df_file = sess_dict['df_file']
    df = pd.read_csv(df_file)

    _, kw = ind.vectorizer(df['summary'], max_features=50, n_max=1, max_df=0.70, min_df=0.01)

    intersect = np.intersect1d(kw, resume_kw)

    return intersect
