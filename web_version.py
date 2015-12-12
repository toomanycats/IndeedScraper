######################################
# File Name : web_version.py
# Author : Daniel Cuneo
# Creation Date : 11-21-2015
######################################
import pdb
import uuid #for random strints
import time
import logging
import pandas as pd
from flask import Flask, request
import indeed_scrape
import jinja2
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.util.string import encode_utf8
from bokeh.charts import Bar
import os
import numpy as np
import json
import pickle

data_dir = os.getenv('OPENSHIFT_DATA_DIR')
if data_dir is None:
    data_dir = os.getenv('PWD')

log_dir = os.getenv('OPENSHIFT_LOG_DIR')
if log_dir is None:
    log_dir = os.getenv("PWD")

logfile = os.path.join(log_dir, 'python.log')
logging.basicConfig(filename=logfile, level=logging.INFO)

session_file = os.path.join(data_dir, 'df_dir', 'session_file.pck')

input_template = jinja2.Template('''
<!DOCTYPE html>
<html lang="en">
<head>
    <title>indeed skill scraper</title>
    <meta charset="UTF-8">
</head>

<body>
        <h1>indeed.com job openings skill scraper</h1>

        <form action="/get_data/"  method="POST">

            <h2>Enter <strong>job title keywords</strong> you normally use to search for openings on indeed.com</h2>
            The scraper will use the "in title" mode of indeed's search engine. Care has been<br>
            taken to not allow duplicates. Depending on how many posting you enter, it can take<br>
            a long time to complete. Start with the default then go higher.<br>
            <input type="text" name="kw" placeholder="data science"><br><br>

             The number of job postings to scrape.<br>
            <input type="text" name="num" value="50"><br><br>

            For now, the zipcodes are regular expression based. If you don't know what that means<br>
            use the default below. This default will search zipcodes that begin with a 9 and a 0,<br>
            which is East and West coasts.<br>
            <input type="text" name="zipcodes" value="^[90]"><br><br>

            Enter where you are searching a known job title or want to use keywords.<br>
            Keywords somteims runs faster though have diverse job title returns.<br>
            Try both keyword and title search to get a diverse idea of the skills<br>
            being sought after.<br>

            <select name="type_">
                <option value='title'>title</option>
                <option value='keywords'>Keywords</option>
            </select>
            <br><br>

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

    <script type="text/javascript">
        $(document).ready(function() {
            $("#chart").load("/run_analysis")
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

    <form  id=radius action="/radius/"  method="post">
        Explore around the radius of a word across all posts. The default is five words in front and in back. <br>
        <input type="text" name="word" placeholder="experience"><br>
        <input type="submit" value="Submit" name="submit">
    </form>

    <h1>Keyword Frequency of Bigrams</h1>
    <div id="chart">Collecting data could take several minutes...</div>

</body>
</html>
""")

radius_template = jinja2.Template('''
<!DOCTYPE html>
<html lang="en-US">
<head>
    <title>radius</title>
    <meta charset="UTF-8">
    <link href="http://cdn.pydata.org/bokeh/release/bokeh-0.9.0.min.css"
          rel="stylesheet" type="text/css">

    <script src="http://cdn.pydata.org/bokeh/release/bokeh-0.9.0.min.js"></script>
</head>

<html>
<body>
    <br><br><br>
    <h2>Words found about a 5 word radius.</h2>

    {{ div }}
    {{ script }}

</body>
</html>
''')

app = Flask(__name__)

def plot_fig(df, num, kws):

    title_string = "Analysis of %i Postings for:'%s'" % (num, kws.strip())

    p = Bar(df, 'kw',
            values='count',
            title=title_string,
            title_text_font_size='15',
            color='blue',
            xlabel="",
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

        type_ = request.form['type_']
        kws = request.form['kw']
        zips = request.form['zipcodes']
        num_urls = int(request.form['num'])

        df_file = os.path.join(data_dir,  'df_dir', mk_random_string())
        logging.info("session file path: %s" % df_file)


        put_to_sess({'type_':type_,
                     'kws':kws,
                     'zips':zips,
                     'num_urls':num_urls,
                     'df_file':df_file
                     })

        logging.info("df file path: %s" % df_file)
        logging.info("type:%s" %  type_)
        logging.info("key words:%s" % kws)
        logging.info("zipcode regex:%s" % zips)
        logging.info("number urls:%s" % num_urls)

        html = output_template.render()

        return encode_utf8(html)

    except Exception, err:
        logging.error(err)
        raise

def get_plot_comp(kw, count, df, title_key):
        count = count.toarray().sum(axis=0)

        num = df['url'].count()

        dff = pd.DataFrame()
        dff['kw'] = kw
        dff['count'] = count

        kws = get_sess()['kws']

        p = plot_fig(dff, num, kws)
        script, div = components(p)

        return script, div

@app.route('/run_analysis/')
def run_analysis():
    try:
        pdb.set_trace()
        logging.info("starting run_analysis %s" % time.strftime("%H:%M:%S") )
        ind = indeed_scrape.Indeed(query_type=get_sess()['type_'])
        ind.query = get_sess()['kws']
        ind.stop_words = "and"
        ind.add_loc = get_sess()['zips']
        ind.num_samp = 1000 # num additional random zipcodes
        ind.num_urls = int(get_sess()['num_urls'])
        ind.main()

        df = ind.df

        # save df for additional analysis
        df.to_csv(get_sess()['df_file'], index=False, encoding='utf-8')

        titles = df['jobtitle'].unique().tolist()
        list_of_titles = '<br>'.join(titles)

        count, kw = ind.vectorizer(df['summary'], n_min=2, n_max=2, max_features=50)
        script, div = get_plot_comp(kw, count, df, 'kws')

        # plot the cities
        df_city = pd.DataFrame({'kw':df['city'], 'count':df['city'].count()})
        cities_p = plot_fig(df_city, df_city.shape[0] , 'Count of Cities in the Analysis.')
        city_script, city_div = components(cities_p)

        output = """
%(kw_script)s
%(kw_div)s

%(cities_script)s
%(cities_div)s

%(titles)s
"""
        output = output %{'kw_script':script,
                          'kw_div':div,
                          'cities_script':city_script,
                          'cities_div':city_div,
                          'titles':list_of_titles
                          }

    except Exception, err:
        logging.info("error: %s" % err)
        return err

@app.route('/radius/', methods=['post'])
def radius():

    kw = request.form['word']
    logging.info("radius key word:%s" % kw)

    df = pd.read_csv(get_sess()['df_file'])
    series = df['summary']
    ind = indeed_scrape.Indeed('kw')
    ind.stop_words = "and"
    ind.add_stop_words()

    words = ind.find_words_in_radius(series, kw, radius=5)
    try:
        count, kw = ind.vectorizer(words, max_features=50, n_min=1, n_max=2)
    except ValueError:
        return "The key word was not found in the top 50."

    script, div = get_plot_comp(kw, count, df, 'radius_kw')
    return radius_template.render(div=div, script=script)

def mk_random_string():
    random_string = str(uuid.uuid4()) + ".csv"

    return random_string

def put_to_sess(values):
    pickle.dump(values, open(session_file, 'wb'))

def get_sess():
    return pickle.load(open(session_file, 'rb'))


if __name__ == "__main__":
    app.run()



