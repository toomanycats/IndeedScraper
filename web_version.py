######################################
# File Name : web_version.py
# Author : Daniel Cuneo
# Creation Date : 11-21-2015
######################################
import time
import logging
import pandas as pd
from flask import Flask, request, session
import indeed_scrape
import jinja2
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.util.string import encode_utf8
from bokeh.charts import Bar
import os
import numpy as np
import ConfigParser

data_dir = os.getenv('OPENSHIFT_DATA_DIR')
logfile = os.path.join(data_dir, 'logfile.log')
logging.basicConfig(filename=logfile, level=logging.INFO)

repo_dir = os.getenv('OPENSHIFT_REPO_DIR')
config = ConfigParser.RawConfigParser()
config.read(os.path.join(repo_dir, 'tokens.cfg'))
key = config.get("sess_key", 'key')

input_template = jinja2.Template('''
<!DOCTYPE html>
<html lang="en">
<head>
    <title>indeed skill scraper</title>
    <meta charset="UTF-8">
</head>

<body>
        <h3>INDEED.COM JOB OPENINGS SKILL SCRAPER</h3>
        <form action="/get_data/"  method="POST">
            Enter keywords you normally use to search for openings on indeed.com<br>
            <input type="text" name="kw" placeholder="data science"><br>
            Enter zipcodes<br>
            <input type="text" name="zipcodes" value="^[90]"><br>
            <input type="submit" value="Submit" name="submit">
        </form>
</body>
</html>''')

output_template = jinja2.Template("""
<!DOCTYPE html>
<html lang="en-US">
<head>
    <title>indeed skill scraper results</title>
    <meta charset="UTF-8">
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.11.3/jquery.min.js"></script>

    <script>
        $(function () {
        $("#chart").load("/run_analysis");
        });
    </script>
</head>

<link
    href="http://cdn.pydata.org/bokeh/release/bokeh-0.9.0.min.css"
    rel="stylesheet" type="text/css">

<script
    src="http://cdn.pydata.org/bokeh/release/bokeh-0.9.0.min.js"
></script>


<body>

    <h1>Keyword Frequency of Stemmed Bigrams</h1>
    <div id="chart">Collecting data may take a while...</div>

</body>
</html>
""")

app = Flask(__name__)
app.secret_key=key

def plot_fig(df, num):
    kws = session['kws']
    zips = session['zips']

    df.sort('count', inplace=True)

    title_string = "Analysis of %i Postings for:'%s'" % (num, kws)

    p = Bar(df, 'kw',
            values='count',
            title=title_string,
            title_text_font_size='15',
            color='blue',
            xlabel="keywords",
            ylabel="Count",
            width=1100,
            height=500)

    return p

@app.route('/')
def get_keywords():
    logging.info("running app:%s" % time.strftime("%d-%m-%Y:%H:%M:%S"))
    return input_template.render()

@app.route('/get_data/', methods=['post'])
def get_data():
    try:
        logging.info("starting get_data: %s" % time.strftime("%H:%M:%S"))
        kws = request.form['kw']
        zips = request.form['zipcodes']

        session['kws'] = kws
        session['zips'] = zips

        logging.info(session['kws'])
        logging.info(session['zips'])

        html = output_template.render()

        return encode_utf8(html)

    except Exception, err:
        logging.error(err)
        raise

@app.route('/run_analysis/')
def run_analysis(num_urls):
    try:
        logging.info("starting run_analysis %s" % time.strftime("%H:%M:%S") )
        ind = indeed_scrape.Indeed()
        ind.query = session['kws']
        ind.stop_words = "and"
        ind.add_loc = session['zips']
        ind.num_samp = 0 # num additional random zipcodes
        ind.num_urls = num_urls
        ind.main()

        df = ind.df
        df = df.drop_duplicates(subset=['url']).dropna(how='any')

        count, kw = ind.vectorizer(df['summary_stem'], n_min=2, max_features=30)
        #convert from sparse matrix to single dim np array
        count = count.toarray().sum(axis=0)

        num = df['url'].count()

        dff = pd.DataFrame()
        dff['kw'] = kw
        dff['count'] = count

        p = plot_fig(dff, num)
        script, div = components(p)
        return "%s\n%s" %(script, div)
        #html = output_template.render(script=script, div=div)
        #return encode_utf8(html)

    except ValueError:
        logging.info("vectorizer found no words")
        html = output_template.render(script="", div="no results")
        return html

    except Exception, err:
        print err
        logging.error(err)
        raise


if __name__ == "__main__":
    app.debug = False
    app.run()
