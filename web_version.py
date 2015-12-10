######################################
# File Name : web_version.py
# Author : Daniel Cuneo
# Creation Date : 11-21-2015
######################################
import uuid #for random strints
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

data_dir = os.getenv('OPENSHIFT_DATA_DIR')
logfile = os.path.join(data_dir, 'logfile.log')
logging.basicConfig(filename=logfile, level=logging.DEBUG)

repo_dir = os.getenv('OPENSHIFT_REPO_DIR')

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
            Keywords typically runs much faster though have diverse job title returns.<br>

            <input type="text" name="type_" value="title"><br><br>

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

</head>

<link
    href="http://cdn.pydata.org/bokeh/release/bokeh-0.9.0.min.css"
    rel="stylesheet" type="text/css">

<script
    src="http://cdn.pydata.org/bokeh/release/bokeh-0.9.0.min.js"
></script>


<body>

    <script type="text/javascript">
        $(function() {
            $("#chart").load("/run_analysis")
            });
    </script>

    <h1>Keyword Frequency of Bigrams</h1>
    <div id="chart">Collecting data could take several minutes...</div>

    <br><br><br>
    <form  id=radius action="/radius/"  method="post">
        Explore around the radius of a word across all posts. The default is five words in front and in back. <br>
        <input type="text" name="word" placeholder="experience"><br>
        <input type="submit" value="Submit" name="submit">
    </form>

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
app.secret_key = str(uuid.uuid4())

def plot_fig(df, num, kws):

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
    app.secret_key = str(uuid.uuid4())
    logging.info("running app:%s" % time.strftime("%d-%m-%Y:%H:%M:%S"))
    df_file = os.path.join(data_dir,  'df_dir', mk_random_string())

    logging.info("df file path: %s" % df_file)
    session['df_file'] = df_file

    return input_template.render()

@app.route('/get_data/', methods=['post'])
def get_data():
    try:
        logging.info("starting get_data: %s" % time.strftime("%H:%M:%S"))
        type_ = request.form['type_']
        kws = request.form['kw']
        zips = request.form['zipcodes']
        num_urls = int(request.form['num'])

        session['kws'] = kws
        session['zips'] = zips
        session['num_urls'] = num_urls
        session['type_'] = type_

        logging.info("type:%s" % session['type_'])
        logging.info("key words:%s" % session['kws'])
        logging.info("zipcode regex:%s" % session['zips'])
        logging.info("number urls:%s" % session['num_urls'])

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

        kws = session[title_key]

        p = plot_fig(dff, num, kws)
        script, div = components(p)

        return script, div

@app.route('/run_analysis/')
def run_analysis():
    try:
        logging.info("starting run_analysis %s" % time.strftime("%H:%M:%S") )
        ind = indeed_scrape.Indeed(query_type=session['type_'])
        ind.query = session['kws']
        ind.stop_words = "and"
        ind.add_loc = session['zips']
        ind.num_samp = 0 # num additional random zipcodes
        ind.num_urls = int(session['num_urls'])
        ind.main()

        df = ind.df

        # save df for additional analysis
        df.to_csv(session['df_file'], index=False)
        # save titles for later
        titles = df['jobtitle'].unique().tolist()

        count, kw = ind.vectorizer(df['summary'], n_min=2, max_features=50)

        list_of_titles = '<br>'.join(titles)

        script, div = get_plot_comp(kw, count, df, 'kws')
        return "%s\n%s\n\n%s" %(script, div, list_of_titles)

    except ValueError:
        logging.info("vectorizer found no words")
        return "no keywords found"

    except Exception, err:
        print err
        logging.error(err)
        raise

@app.route('/radius/', methods=['post'])
def radius():

    kw = request.form['word']
    session['radius_kw'] = kw
    logging.info("radius key word:%s" % kw)

    df = pd.read_csv(session['df_file'])
    series = df['summary']
    ind = indeed_scrape.Indeed('kw')
    ind.stop_words = "and"
    ind.add_stop_words()

    words = ind.find_words_in_radius(series, kw, radius=5)
    try:
        count, kw = ind.vectorizer(words, max_features=50, n_min=1)
    except ValueError:
        return "The key word was not found in the top 50."

    script, div = get_plot_comp(kw, count, df, 'radius_kw')
    return radius_template.render(div=div, script=script)

def mk_random_string():
    random_string = str(uuid.uuid4()) + ".csv"

    return random_string


if __name__ == "__main__":
    app.run()



