######################################
# File Name : web_version.py
# Author : Daniel Cuneo
# Creation Date : 11-21-2015
######################################
import ConfigParser
import time
import logging
import pandas as pd
from flask import Flask, request, render_template, session
import indeed_scrape
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.util.string import encode_utf8
from bokeh.charts import Bar
from bokeh.resources import INLINE
import os

data_dir = os.getenv('OPENSHIFT_DATA_DIR')
logfile = os.path.join(data_dir, 'logfile.log')
logging.basicConfig(filename=logfile, level=logging.INFO)

repo_dir = os.getenv("OPENSHIFT_REPO_DIR")
config_path = os.path.join(repo_dir, "tokens.cfg")

config_parser = ConfigParser.RawConfigParser()
config_parser.read(config_path)
sess_key = config_parser.get("flask_session_key", "key")

app = Flask(__name__)
app.secret_key = sess_key


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
    try:
        logging.info("running app:%s" % time.strftime("%d-%m-%Y:%H:%M:%S"))
        return render_template('index.html')

    except Exception, err:
        logging.error(err)
        raise

@app.route('/get_data/', methods=['get', 'post'])
def get_data():
    try:
        logging.info("getting form data: %s" % time.strftime("%H:%M:%S"))
        session['kws'] = request.form['kw']
        session['zips'] = request.form['zipcodes']

        logging.info(session['kws'])
        logging.info(session['zips'])

        script, div = run_analysis()

        js_resources = INLINE.render_js()
        css_resources = INLINE.render_css()

        html = render_template('output.html',
                                plot_script=script,
                                plot_div=div,
                                js_resources=js_resources,
                                css_resources=css_resources
                                )

        return encode_utf8(html)

    except Exception, err:
        logging.error(err)
        raise

def run_analysis(num_urls=100):
    try:
        logging.info("starting run_analysis %s" % time.strftime("%H:%M:%S") )
        ind = indeed_scrape.Indeed()
        ind.query = session['kws']
        ind.stop_words = "and"
        ind.add_loc = session['zips']
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
        script, div = components(p, INLINE)

        return script, div

    except ValueError:
        logging.info("vectorizer found no words")
        html = render_template('output.html', script="no results found" )

    except Exception, err:
        logging.error(err)
        raise


if __name__ == "__main__":
    app.debug = False
    app.run()
