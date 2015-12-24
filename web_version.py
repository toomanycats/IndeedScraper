#####################################
# File Name : web_version.py
# Author : Daniel Cuneo
# Creation Date : 11-21-2015
######################################
from fuzzywuzzy import fuzz
import sqlalchemy
import uuid #for random strints
import subprocess
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
import compare
import pdb

sql_username = os.getenv("OPENSHIFT_MYSQL_DB_USERNAME")
if sql_username is None:
    sql_username = 'root'

sql_password = os.getenv("OPENSHIFT_MYSQL_DB_PASSWORD")
if sql_password is None:
    sql_password = 'test'

mysql_ip = os.getenv("OPENSHIFT_MYSQL_DB_HOST")
if mysql_ip is None:
    mysql_ip = '127.0.0.1'

repo_dir = os.getenv("OPENSHIFT_REPO_DIR")
if repo_dir is None:
    repo_dir = os.getenv("PWD")

data_dir = os.getenv('OPENSHIFT_DATA_DIR')
if data_dir is None:
    data_dir = os.getenv('PWD')

log_dir = os.getenv('OPENSHIFT_LOG_DIR')
if log_dir is None:
    log_dir = os.getenv("PWD")

missing_keywords = compare.MissingKeywords()

logfile = os.path.join(log_dir, 'python.log')
logging.basicConfig(filename=logfile, level=logging.DEBUG)

session_file = os.path.join(data_dir, 'df_dir', 'session_file.pck')

missing_template = jinja2.Template('''
<!DOCTYPE html>
<html lang="en">
<head>
    <link rel="stylesheet" href="css/style.css">
    <title>missing keyword analysis</title>
    <meta charset="UTF-8">
    <style>
        p {
            margin: 0.2cm 0.5cm 0.1cm 1cm;
            font-family:"Verdana";
            font-size:110%
        }

        table {
            border-collapse: collapse;
            width: 100%;
        }

        th, td {
            text-align: left;
            padding: 8px;
        }

        tr:nth-child(even){background-color: #f2f2f2}

        th {
            background-color: #4CAF50;
            color: white;
        }
    </style>

</head>
<body>

<h1>Upload your resume for an analysis of missing keywords compared with the
job search results.</h1>

<p>This service will extract the text from your resume and compare it to the list of keywords found in the previous analysis. The output will be the keywords not included in your resume that were found in the job postings.</p>

<form action='/missing/' method=POST enctype=multipart/form-data>
    <input type=file name=File>
    <input type=submit value=Upload>
</form>

<br><br>

<h2>Comparison of your resume with the single keyword analysis</h2>
    <table>
    <tr>
        <th>Missing Keywords in Your Resume</th>
    </tr>
        {{ rows }}
    </table>

</body>
</html>
''')

grammar_template = jinja2.Template('''
<!DOCTYPE html>
<html lang="en">
<head>
    <title>analysis of full job posting text</title>
    <meta charset="UTF-8">
    <style>
        body {
            background-color: #caf6f6;
            }
    </style>
    <link href="http://cdn.pydata.org/bokeh/release/bokeh-0.9.0.min.css"
          rel="stylesheet" type="text/css">

    <script src="http://cdn.pydata.org/bokeh/release/bokeh-0.9.0.min.js"></script>
</head>

<body>
<h1>General Language Analysis</h1> <p>The previous analysis of single and
double keywords was focused on skills. This treatment tries to avoid
the bulleted skills and find meaning in the general text.</p>

<br>
<p><i>The graph is interactive, scroll up and down to zoom</i></p>

{{ div }}
{{ script }}

</body>
</html>
''')

cities_template = jinja2.Template('''
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Count of job Postings per City: Showing Top 20 Citites</title>
    <style>
        body {
            background-color: #caf6f6;
            }
    </style>
    <meta charset="UTF-8">
    <link href="http://cdn.pydata.org/bokeh/release/bokeh-0.9.0.min.css"
          rel="stylesheet" type="text/css">

    <script src="http://cdn.pydata.org/bokeh/release/bokeh-0.9.0.min.js"></script>
</head>

<body>
<h1>Count of job Postings per City: Showing Top 20 Citites</h1>
{{ div }}
{{ script }}
</body>
</html>
''')

title_template = jinja2.Template('''
<!DOCTYPE html>
<html lang="en">
<head>
    <link rel="stylesheet" href="css/style.css">
    <style>
        p {
            margin: 0.2cm 0.5cm 0.1cm 1cm;
            font-family:"Verdana";
            font-size:110%
        }

        li {
        padding-left: 2cm;
        font-size:110%
        }

        table {
            border-collapse: collapse;
            width: 100%;
        }

        th, td {
            text-align: left;
            padding: 8px;
        }

        tr:nth-child(even){background-color: #f2f2f2}

        th {
            background-color: #4CAF50;
            color: white;
        }
    </style>
</head>

<body>
    <p>The table below is a list of the job titles that formed the search
    results. This table can provide some insights:</p>

    <p>More unique titles is a measure of how your keywords/title search terms, track across domains.</p>
    <li> A zero count for a title, means that the job posting(s) were not analyzed
    due to formatting concerns. However, the title was found in the search.</li>
    <li>All titles are reduced to lower case and fuzzy matched, so that very similar titles are grouped together</li>

    <table>
    <tr>
        <th>Job Title From Posting</th><th>Count</th>
    </tr>
        {{ rows }}
    </table>

</body>
</html>
''')

stem_template= jinja2.Template('''
<!DOCTYPE html>
<html lang="en">
<head>
    <link rel="stylesheet" href="css/style.css">
    <style>
        p {
            margin: 0.5cm 0.5cm 0.2cm 6cm;
            font-family:"Verdana";
            font-size:150%
        }

        li {
        padding-left: 8cm;
        font-size:150%
        }

        body {
            background-color: #caf6f6;
        }
    </style>
    <title>stemmed results</title>
    <meta charset="UTF-8">
    <link href="http://cdn.pydata.org/bokeh/release/bokeh-0.9.0.min.css"
          rel="stylesheet" type="text/css">

    <script src="http://cdn.pydata.org/bokeh/release/bokeh-0.9.0.min.js"></script>
</head>

<body>
<h1>Frequency of Single Keywords:Skills Focused</h1>
<p><i>The graph is interactive, scroll up and down to zoom</i></p>

{{ div }}

{{ script }}

<p>All the words in the sample have been reduced to their "stems". <br> That
is, the suffixes have been removed,</p>

<li>working</li>
<li>works</li>
<li>worked</li>

<p>are counted the same, as "work".
<li>The bar graph shows only single keywords, known as Mono-grams.</li>
<li>Many words will be reduced, 'requirements', or, 'experience' will look like:
<p>
    <li>requir</li>
    <li>experi</li>
</p>


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
    <title>keyword counter optimizer</title>
    <meta charset="UTF-8">
    <meta name="description" content="Optimize Your Resume and Social Media Keywords, Provide statistics on the keywords used in job postings Group the cities they are from Report the job titles searched." />
    <meata name="robots" content="index, follow" />
    <style>
    p {
        margin: 0.5cm 0.5cm 0.2cm 6cm;
        font-family:"Verdana";
        font-size:150%
      }
    li {
        padding-left: 8cm;
        font-size:150%
       }

    html {
            background: url(static/background.jpg) no-repeat center center fixed;
            -webkit-background-size: cover;
            -moz-background-size: cover;
            -o-background-size: cover;
            background-size: cover;
            color: #e5ffff;
         }

    body {
            background: url('static/med_dir_300.png') no-repeat fixed;
            -webkit-background-size: contain;
            -moz-background-size: contain;
            -o-background-size: contain;
            background-size: contain;
          }
    </style>

</head>

<body>
        <center><img src=static/dir_marketing.png alt="Sample Keyword Output" style="width:1200px;height:425px;"></center><br>
        <center><h1>Optimize Your Keywords for Resumes and LinkedIn</h1></center>

        <form action="/get_data/"  method="POST">

        <p>Enter your keywords here <input type="text" name="kw" placeholder="data science" spellcheck="true"></p>

            <p>The number of job postings to examine</p>

            <p><select name="num">
                <option value=10>10 primarily for testing</option>
                <option value=50>50</option>
                <option value=100>100</option>
                <option value=200>200</option>
                </select>
            </p>

            <p>Select whether you want to use the keyword or title mode. I suggest
            trying out both. The uniqued list of job titles will be
            displayed with the results so that you can determine if the
            keywords were appropriate or not. You also learn something about
            the fields  which use the skills you are providing as search
            terms.</p>

            <p><select name="type_">
                <option value='title'>title</option>
                <option value='keywords'>keywords</option>
                </select>
            </p>

            <p><input type="submit" value="Submit" name="submit"></p>
        </form>

<!-- Start of StatCounter Code for Default Guide -->
<script type="text/javascript">
var sc_project=10739395;
var sc_invisible=1;
var sc_security="0d075499";
var scJsHost = (("https:" == document.location.protocol) ?
"https://secure." : "http://www.");
document.write("<sc"+"ript type='text/javascript' src='" +
scJsHost+
"statcounter.com/counter/counter.js'></"+"script>");
</script>
<noscript><div class="statcounter"><a title="free web stats"
href="http://statcounter.com/" target="_blank"><img
class="statcounter"
src="http://c.statcounter.com/10739395/0/0d075499/1/"
alt="free web stats"></a></div></noscript>
<!-- End of StatCounter Code for Default Guide -->

</body>
</html>''')

output_template = jinja2.Template("""
<!DOCTYPE html>
<html lang="en-US">
<head>
    <title>keyword counter results</title>
    <meta charset="UTF-8">
    <style>
        body {
            background-color: #caf6f6;
            font-size: 125%;
            }
    </style>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.11.3/jquery.min.js"></script>

    <script type="text/javascript">
        $(function() {
            $("#chart").load("/run_analysis", function() {
                $("#stem").slideToggle("fast", function() {
                    $("#grammar").slideToggle("/grammar/", function() {
                        $("#cities").slideToggle("fast", function() {
                            $("#titles").slideToggle("fast", function() {
                                $("#radius").slideToggle("fast", function() {
                                    $("#missing").slideToggle("fast");
                                    });
                                });
                            });
                        });
                    });
                });
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

    <h1>Frequency of Keyword Pairs</h1>
    <p><i>The graph is interactive, scroll up and down to zoom</i></p>
    <div id="chart">Collecting data might take a minute...
    <p>This analysis uses all the text found in the bullet points. Typically,
    these are where the hard skills are listed for the applicant.</p>
    </div>

    <br><br>

    <div id=stem style="display: none">
    <p>Also text from bullet points, the analysis below produces single words.</p>
    <a href="/stem/">Single Word Analysis</a>
    </div>

    <br>


    <div id=grammar style="display: none">
    <p>Not all interesting keywords are contained in the bullet points. This
    treatment will search the body of the post not including bulleted lists.
    You may find more soft skills here.</p>
    <a href="/grammar/">Alternate Skill Analysis</a>
    </div>

    <br><br>

    <div id=cities style="display: none">
    <p>A count of the job postings per city.</p>
    <a href="/cities/"> Show Cities </a>
    </div>

    <br>

    <div id=titles style="display: none">
    <p> A break down of the job titles returned. This is most useful when using
    a keyword search on indeed, to make sure your keywords are matching the jobs you
    want.</p>
    <a href="/titles/"> Show Job Titles </a>
    </div>

    <br>

    <form  id=radius action="/radius/"  method="post" style="display: none">
        Explore around the radius of a word across all posts.<br>
        The default is five words in front and in back. <br>
        <input type="text" name="word" placeholder="experience"><br>
        <input type="submit" value="Submit" name="submit">
    </form>

    <br>

    <div id=missing style="display: none">
    <p>Upload a PDF copy of your resume, and it will be compared with the job
    posting keyword anaysis.</p>
    <a href="/missing/"> Analyze Your Resume For Missing Keywords</a>
    </div>

</body>
</html>
""")

radius_template = jinja2.Template('''
<!DOCTYPE html>
<html lang="en-US">
<head>
    <style>
        body {
            background-color: #caf6f6;
            }
    </style>
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
    <p><i>The graph is interactive, scroll up and down to zoom</i></p>

    {{ div }}
    {{ script }}

</body>
</html>
''')

app = Flask(__name__)

stop_words = 'religion sex disibility veteran status sexual orientation and work ability http https www gender'

def plot_fig(df, num, kws):

    title_string = "Analysis of %i Postings for:'%s'" % (num, kws.strip())

    p = Bar(df, 'kw',
            values='count',
            title=title_string,
            title_text_font_size='20',
            color='blue',
            xlabel="",
            ylabel="Number of Posts Containing Keyword",
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
    kws = request.form['kw'].lower()
    num_urls = int(request.form['num'])

    df_file = os.path.join(data_dir,  'df_dir', mk_random_string())
    logging.info("session file path: %s" % df_file)

    put_to_sess({'type_':type_,
                 'kws':kws,
                 'num_urls':num_urls,
                 'df_file':df_file,
                 })

    logging.info("running get_data:%s" % time.strftime("%d-%m-%Y:%H:%M:%S"))
    logging.info("df file path: %s :%s" % (df_file, time.strftime("%d-%m-%Y:%H:%M:%S")))
    logging.info("type:%s" %  type_)
    logging.info("key words:%s" % kws)
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

@app.route('/titles/')
def plot_titles():
    df_file = get_sess()['df_file']
    df = load_csv()

    df['jobtitle'] = df['jobtitle'].apply(lambda x:x.lower())
    ind = indeed_scrape.Indeed
    title_de_duped = ind.summary_similarity(df, 'jobtitle', 80)

    grp = title_de_duped.groupby("jobtitle").count().sort("url", ascending=False)
    cities = grp.index.tolist()
    counts = grp['city'] # any key will do here

    row = u'<tr><td>%s</td><td>%s</td></tr>'
    rows = u''

    for cty, cnt in zip(cities, counts):
        try:
            cty = cty.encode("utf-8", "ignore")
            rows += row % (cty, cnt)
        except:
           continue

    page = title_template.render(rows=rows)
    return encode_utf8(page)

@app.route('/cities/')
def plot_cities():
    df = load_csv()

    count = df.groupby("city").count()['url']
    cities = count.index.tolist()
    num_posts = df.shape[0]

    df_city = pd.DataFrame({'kw':cities, 'count':count})
    df_city.sort('count', ascending=False, inplace=True)
    df_city.reset_index(inplace=True)

    n = df_city.shape[0]
    if n > 20:
        end = 20
    else:
        end = n

    p = plot_fig(df_city.loc[0:end,:], num_posts, 'Posts Per City in the Analysis:Top 20')
    script, div = components(p)

    page = cities_template.render(div=div, script=script)

    return encode_utf8(page)
@app.route('/run_analysis/')
def run_analysis():
    sess_dict = get_sess()
    logging.info("starting run_analysis %s" % time.strftime("%H:%M:%S") )
    ind = indeed_scrape.Indeed(query_type=sess_dict['type_'])
    ind.query = sess_dict['kws']
    ind.stop_words = stop_words
    ind.num_urls = int(sess_dict['num_urls'])

    ind.main()
    df = ind.df

    # save df for additional analysis
    save_to_csv(df)
    to_sql()

    count, kw = ind.vectorizer(df['summary'], n_min=2, n_max=2, max_features=60,
            max_df=compute_max_df(sess_dict['type_'], sess_dict['num_urls']))

    script, div = get_plot_comp(kw, count, df, 'kws')


    output = """
%(kw_div)s
%(kw_script)s
"""
    return output %{'kw_script':script,
                    'kw_div':div
                    }

@app.route('/radius/', methods=['post'])
def radius():
    sess_dict = get_sess()
    kw = request.form['word']
    logging.info("radius key word:%s" % kw)

    df = load_csv()
    ind = indeed_scrape.Indeed('kw')
    ind.stop_words = stop_words
    ind.add_stop_words()
    ind.df = df

    words = ind.build_corpus_from_sent(kw)
    words = ind.find_words_in_radius(words, kw, 5)

    try:
        count, kw = ind.vectorizer(words, max_features=50, n_min=1, n_max=2,
               max_df=compute_max_df(sess_dict['type_'], sess_dict['num_urls']))
    except ValueError:
        return "The body of words compiled did not contain substantially repeated terms."

    script, div = get_plot_comp(kw, count, df, 'radius_kw')
    return radius_template.render(div=div, script=script)

@app.route('/stem/')
def stem():
    logging.info("running stem")
    sess_dict = get_sess()
    df_file = sess_dict['df_file']
    df = load_csv()
    summary_stem = df['summary_stem']

    ind = indeed_scrape.Indeed("kw")
    ind.stop_words = stop_words
    ind.add_stop_words()

    count, kw = ind.vectorizer(summary_stem, n_min=1, n_max=2, max_features=80,
            max_df=compute_max_df(sess_dict['type_'], sess_dict['num_urls']))
    script, div = get_plot_comp(kw, count, df, 'kws')

    page = stem_template.render(script=script, div=div)
    return encode_utf8(page)

@app.route('/grammar/')
def grammar_parser():
    logging.info("running grammar parser")
    sess_dict = get_sess()
    df = load_csv()
    df.dropna(subset=['grammar'], inplace=True)

    ind = indeed_scrape.Indeed('kw')
    ind.stop_words = stop_words
    ind.add_stop_words()

    count, kw = ind.tfidf_vectorizer(df['grammar'], n_min=2, n_max=3, max_features=30,
            max_df=0.75, min_df=0.01)

    script, div = get_plot_comp(kw, count, df, 'kws')

    page = grammar_template.render(script=script, div=div)
    return encode_utf8(page)

@app.route('/missing/', methods=['GET', 'POST'])
def compute_missing_keywords():
    if request.method == "POST":
        resume_file = request.files['File']
        logging.info("resume path: %s" % resume_file.filename)

        resume_path = os.path.join(data_dir, resume_file.filename)
        resume_file.save(resume_path)

        df = load_csv()
        rows = missing_keywords.main(resume_path, df['summary'])

        _gzip(resume_path)

        return missing_template.render(rows=rows)

    else:
        return missing_template.render()

def mk_random_string():
    random_string = str(uuid.uuid4()) + ".csv"

    return random_string

def put_to_sess(values):
    pickle.dump(values, open(session_file, 'wb'))

def get_sess():
    return pickle.load(open(session_file, 'rb'))

def compute_max_df(type_, num_samp):
    if type_ == 'keywords':
        base = 0.80

    elif type_ == 'title':
        base = 0.85

    else:
        raise ValueError, "type not understood"

    if num_samp < 50:
        return base + 0.10

    else:
        return base

def to_sql():
    sess_dict = get_sess()

    norm_keywords = indeed_scrape.Indeed._split_on_spaces(sess_dict['kws'])
    norm_keywords = " ".join(norm_keywords)
    reference = pd.DataFrame({'keyword':norm_keywords,
                              'df_file':sess_dict['df_file']
                             }, index=[0])

    conn_string = "mysql://%s:%s@%s/indeed" %(sql_username, sql_password, mysql_ip)
    sql_engine = sqlalchemy.create_engine(conn_string)
    reference.to_sql(name='data', con=sql_engine, if_exists='append', index=False)

def load_csv():
    sess_dict = get_sess()
    df_file = _ungzip(sess_dict['df_file'])
    df = pd.read_csv(df_file)

    _gzip(sess_dict['df_file'])

    return df

def save_to_csv(df):
    sess_dict = get_sess()
    df.to_csv(sess_dict['df_file'], index=False, quoting=1, encoding='utf-8')
    df_file =  _gzip(sess_dict['df_file'])

    #re write the pickle file with gz ext for df_file
    sess_dict['df_file'] = df_file
    put_to_sess(sess_dict)

def _call_shell(cmd):
    logging.debug('command line used: %s' %cmd)

    process = subprocess.Popen(cmd, shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)

    out, err = process.communicate()
    errcode = process.returncode

    logging.debug('Shell call output: %s' %out)

    return out

def _ungzip(File, force=True):
    if force is True:
        cmd = 'gunzip -f %s' %File
    else:
        cmd = 'gunzip %s' %File

    _call_shell(cmd)

    return File.replace('.gz', '')

def _gzip(File):
    cmd = 'gzip -f %s' %File
    _call_shell(cmd)

    return File + '.gz'


if __name__ == "__main__":
    app.run()
