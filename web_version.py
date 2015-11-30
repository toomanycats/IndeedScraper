######################################
# File Name : web_version.py
# Author : Daniel Cuneo
# Creation Date : 11-21-2015
######################################
import time
import logging
import pandas as pd
from flask import Flask
from flask import request, render_template
import indeed_scrape
import jinja2
from bokeh.embed import components
from bokeh.plotting import figure, output_file
from bokeh.util.string import encode_utf8
from bokeh.charts import Bar
import os
import numpy as np

data_dir = os.getenv('OPENSHIFT_DATA_DIR')
logfile = os.path.join(data_dir, 'logfile.log')
logging.basicConfig(filename=logfile, level=logging.INFO)

input_template = jinja2.Template('''
<!DOCTYPE html>
<html lang="en">
<head>
<script type="text/javascript">// <![CDATA[
        function preloader(){
            document.getElementById("loading").style.display = "none";
            document.getElementById("content").style.display = "block";
        }//preloader
        window.onload = preloader;
// ]]></script>

    <title>indeed skill scraper</title>
    <meta charset="UTF-8">

    <style type="text/css">
        div#content {
        display: none;
        }

        div#loading {
            margin: auto;
            width: 35px;
            height: 35px;
            z-index: 1000;
            display: none;
            background: url(/static/loading.gif) no-repeat;
            cursor: wait;
            }
    </style>
</head>

<body>
    <div id="loading"></div>

    <div id="content">
        <h3>INDEED.COM JOB OPENINGS SKILL SCRAPER</h3>
        <form action="/"  method="POST">
            Enter keywords you normally use to search for openings on indeed.com<br>
            <input type="text" name="kw" placeholder="data science"><br>
            Enter zipcodes<br>
            <input type="text" name="zipcodes" placeholder="^(941)"><br>
            <input type="submit" value="Submit" name="submit" onclick="loading();">
        </form>
    </div>
</body>
</html>''')

output_template = jinja2.Template("""
<!DOCTYPE html>
<html lang="en-US">
<head>
    <title>indeed skill scraper results</title>
    <meta charset="UTF-8">
</head>

<link
    href="http://cdn.pydata.org/bokeh/release/bokeh-0.9.0.min.css"
    rel="stylesheet" type="text/css">

<script
    src="http://cdn.pydata.org/bokeh/release/bokeh-0.9.0.min.js"
></script>

<body>
        <h1>INDEED.COM JOB OPENINGS SKILL SCRAPER RESULTS</h1>

        {{ script }}

        {{ div }}
</body>

</html>
""")

app = Flask(__name__)

def plot_fig(df, num):

    title_string = "Analysis of %i Postings" % num

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

@app.route('/', methods=['post'])
def get_data():
    try:
        logging.info("starting get_data: %s" % time.strftime("%H:%M:%S"))
        kws = request.form['kw']
        zips = request.form['zipcodes']
        logging.info(kws)
        logging.info(zips)
        template = run_analysis(kws, zips)

        return template

    except Exception, err:
        logging.error(err)
        raise

def run_analysis(kws, zips, num_urls=200):
    try:
        logging.info("starting run_analysis %s" % time.strftime("%H:%M:%S") )
        ind = indeed_scrape.Indeed()
        ind.query = kws
        ind.stop_words = "and"
        ind.add_loc = zips
        ind.num_samp = 10 # num additional random zipcodes
        ind.num_urls = num_urls# max of 100 postings

        ind.main()
        df = ind.df
        df = df.drop_duplicates(subset=['url']).dropna(how='any')
        count, kw = ind.vectorizer(df['summary_stem'], max_features=50)
        #convert from sparse matrix to single dim np array
        count = count.toarray().sum(axis=0)
        num = ind.df['url'].count()
        dff = pd.DataFrame()
        dff['kw'] = kw
        dff['count'] = count
        p = plot_fig(dff, num)
        script, div = components(p)
        html = output_template.render(script=script, div=div)

        return encode_utf8(html)

    except ValueError:
        logging.info("vectorizer found no words")
        pass

    except Exception, err:
        print err
        logging.error(err)
        raise


if __name__ == "__main__":
    app.debug = False
    app.run()
