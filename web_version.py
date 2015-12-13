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
from flask import Flask, request, redirect, url_for, jsonify
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

stem_template= jinja2.Template('''
<!DOCTYPE html>
<html lang="en">
<head>
    <title>stemmed results</title>
    <meta charset="UTF-8">
    <link href="http://cdn.pydata.org/bokeh/release/bokeh-0.9.0.min.css"
          rel="stylesheet" type="text/css">

    <script src="http://cdn.pydata.org/bokeh/release/bokeh-0.9.0.min.js"></script>
</head>

<body>
{{ div }}

{{ script }}
</body>
</html>
''')

error_template = jinja2.Template('''
<!DOCTYPE html>
<html lang="en">
<head>
    <title>error page</title>
    <meta charset="UTF-8">
</head>

<body>
<strong>
Sorry, but an error occured in the program. <br>
This app is still a work in progress. <br><br>

<li> Please check that your inputs are reasonable. </li>
<li> Sometimes job titles are not found or keywords are not common enough.</li>
</strong>

<br><br>
{{ error }}
<br><br>

</body>
</html>
''')

input_template = jinja2.Template('''
<!DOCTYPE html>
<html lang="en">
<head>
    <title>indeed skill scraper</title>
    <meta charset="UTF-8">
</head>

<body>
        <h1>indeed.com job openings skill scraper</h1> The main purpose of this
        app is to figure out either what skills you need for <br> a given
        field, or, what keywords to use in your resume or LinkedIn
        summary.<br><br>

        There's no real analysis done for you at this point, so be creative
        with what <br> you learn here.<br><br>


        <form action="/get_data/"  method="POST">

            <h2>Enter <strong>job title keywords</strong> you normally use to
            search for openings on indeed.com</h2> The scraper will use the "in
            title" mode or the "keyword" mode, of indeed's search engine. Care
            has been<br>
            taken not to allow duplicate job postings. <br><br>

            Depending on how many posting you enter, it can take a long time to complete. <br>
            Start with the default then go higher.<br><br>

            <input type="text" name="kw" placeholder="data science"><br><br>

            The number of job postings to scrape.<br><br>

            <input type="text" name="num" value="50"><br><br>

            For now, the zipcodes are regular expression based. If you don't
            know what that means<br> use the default below. This default will
            search zipcodes that begin with a 9 or a 0,<br> which is East and
            West coasts.NOTE: a random sampling of 1000 zip codes is also
            included <br> to round out the results. This maynot actually be
            helpful.<br><br>

            <input type="text" name="zipcodes" value="^[90]"><br><br>

            Select whether you want to use the keyword or title mode. I suggest
            trying out <br> both. The uniqued list of job titles will be
            dsiplayed with the results so that you can<br> determine if the
            keywords were appropriate or not. You also learn something about
            the fields <br> which use the skills you are providing as search
            terms.<br><br>


            <select name="type_">
                <option value='title'>title</option>
                <option value='keywords'>Keywords</option>
                <option value='keywords_title'>Keywords and Title</option>
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
            $("#chart").load("/run_analysis", function() {
                $("#stem").slideToggle("slow")
                })
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

    <h1>Keyword Frequency of Bigrams</h1>
    <div id="chart">Collecting data could take several minutes...</div>

    <form  id=radius action="/radius/"  method="post">
        Explore around the radius of a word across all posts.<br>
        The default is five words in front and in back. <br>
        <input type="text" name="word" placeholder="experience"><br>
        <input type="submit" value="Submit" name="submit">
    </form>

    <div id=stem style="display: none">
    <a href="/stem/"> Use Stemmed Keyword </a>
    </div>

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
            width=1500,
            height=500)

    return p

@app.errorhandler(500)
def internal_error(error):
    return error_template.render(error=error)

@app.route('/')
def get_keywords():
    logging.info("running app:%s" % time.strftime("%d-%m-%Y:%H:%M:%S"))
    return input_template.render()

@app.route('/get_data/', methods=['post'])
def get_data():
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

    #pdb.set_trace()
    logging.info("starting run_analysis %s" % time.strftime("%H:%M:%S") )
    ind = indeed_scrape.Indeed(query_type=get_sess()['type_'])
    ind.query = get_sess()['kws']
    ind.stop_words = "and"
    ind.add_loc = get_sess()['zips']
    ind.num_samp = 1000 # num additional random zipcodes
    ind.num_urls = int(get_sess()['num_urls'])
    ind.zip_code_error_limit = 1000
    ind.main()

    df = ind.df

    # save df for additional analysis
    df.to_csv(get_sess()['df_file'], index=False, encoding='utf-8')

    titles = df['jobtitle'].unique().tolist()
    list_of_titles = '<br>'.join(titles)
    list_of_titles += '<br><br>'

    count, kw = ind.vectorizer(df['summary'], n_min=2, n_max=2, max_features=50)
    script, div = get_plot_comp(kw, count, df, 'kws')

    # plot the cities
    count = df.groupby("city").count()['url']# plot fun uses 'kw'
    cities = df['city'].unique()
    df_city = pd.DataFrame({'kw':cities, 'count':count})
    num_posts = df.shape[0]
    cities_p = plot_fig(df_city, num_posts, 'Count of Cities in the Analysis.')
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

    return output

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

@app.route('/stem/')
def stem():
    logging.info("running stem")

    df_file = get_sess()['df_file']
    df = pd.read_csv(df_file)
    summary_stem = df['summary_stem']

    ind = indeed_scrape.Indeed("kw")
    count, kw = ind.vectorizer(summary_stem, n_min=2, n_max=2, max_features=50)
    script, div = get_plot_comp(kw, count, df, 'kws')

    page = stem_template.render(script=script, div=div)
    return encode_utf8(page)

def mk_random_string():
    random_string = str(uuid.uuid4()) + ".csv"

    return random_string

def put_to_sess(values):
    pickle.dump(values, open(session_file, 'wb'))

def get_sess():
    return pickle.load(open(session_file, 'rb'))

if __name__ == "__main__":
    app.run()



