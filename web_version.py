#####################################
# File Name : web_version.py
# Author : Daniel Cuneo
# Creation Date : 11-21-2015
######################################
from MySQLdb import escape_string
from fuzzywuzzy import fuzz
import sqlalchemy
import uuid
import subprocess
import time
import logging
import pandas as pd
from flask import Flask, request, redirect, url_for, jsonify, render_template, session
import indeed_scrape
import jinja2
from bokeh.embed import components
from bokeh.util.string import encode_utf8
from bokeh.charts import Bar
import os
import numpy as np
import json
import pickle
import compare
import pdb
import time
import json

def mk_random_string():
    random_string = str(uuid.uuid4())

    return random_string

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
logging.basicConfig(filename=logfile, level=logging.INFO)

session_file = os.path.join(data_dir, 'df_dir', 'session_file.pck')

conn_string = "mysql://%s:%s@%s/indeed" %(sql_username, sql_password, mysql_ip)

app = Flask(__name__)
app.secret_key = mk_random_string()

stop_words = 'resume affirmative cover letter equal religion sex disibility veteran status sexual orientation and work ability http https www gender com org'

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

@app.route('/gallery/')
def gallery():
    return render_template('gallery.html')

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html',error=error)

@app.route('/')
def get_keywords():
    logging.info("running app:%s" % time.strftime("%d-%m-%Y:%H:%M:%S"))
    return render_template('keywordcounter.html')

@app.route('/get_data/', methods=['GET', 'POST'])
def get_data():
    logging.info("starting get_data: %s" % time.strftime("%H:%M:%S"))

    if request.method == "POST":
        type_ = request.form['type_']
        kws = request.form['kw'].lower()
        kws = indeed_scrape.Indeed._split_on_spaces(kws)
        kws = " ".join(kws) #enter into DB normalized

        session_id = mk_random_string()
        logging.info("session id: %s" % session_id)

        # key used for sql to recover other info
        session['session_id'] = session_id

        to_sql({'session_id':session_id,
                'type_':type_,
                'keyword':kws,
                'ind':0,
                'end':0,
                'count_thres':50
                })

        logging.info("running get_data:%s" % time.strftime("%d-%m-%Y:%H:%M:%S"))
        logging.info("type:%s" %  type_)
        logging.info("key words:%s" % kws)

        html = render_template('output.html')
        return encode_utf8(html)

    if request.method == "GET":
        html = output_template.render()
        return encode_utf8(html)

def get_plot_comp(kw, count, df):
        count = count.toarray().sum(axis=0)

        num = df['url'].count()

        dff = pd.DataFrame()
        dff['kw'] = kw
        dff['count'] = count

        kws = get_sess()['keyword'][0]

        p = plot_fig(dff, num, kws)
        script, div = components(p)

        return script, div

@app.route('/titles/')
def plot_titles():
    df_file = get_sess()['df_file']
    df = load_csv()

    df['jobtitle'] = df['jobtitle'].apply(lambda x:x.lower())
    ind = indeed_scrape.Indeed("kw")
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

    page = render_template('titles.html', rows=rows)
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

    page = render_template('cities.html', div=div, script=script)
    return encode_utf8(page)

@app.route('/bigram/')
def get_bigram_again():
    sess_dict = get_sess()
    html = sess_dict['bigram'][0]
    return render_template("bigram.html", html=html)

def look_up_in_db(kw_string, type_):
    sql = "SELECT df_file FROM data WHERE keyword = '%s' and type_ = '%s';"
    sql = sql %(kw_string, type_)

    sql_engine = sqlalchemy.create_engine(conn_string)
    df = pd.read_sql(sql=sql, con=sql_engine)

    if df.shape[0] == 0:
        return None
    else:
        return df['df_file'][0] # just in case there's more than one

def process_data_in_db(df_file):
    ind = indeed_scrape.Indeed("kw")
    sess_dict = get_sess()

    df = pd.read_csv(df_file)

    # discard output
    map(ind.stemmer_, df['summary'])
    inv = json.dumps(ind.stem_inverse)
    update_sql('stem_inv', inv, 'string')

    html = bigram(df, sess_dict['type_'][0], ind)
    update_sql('bigram', html, 'string')
    update_sql('count_thres', df.shape[0] + 20, 'int')

    return html

@app.route('/check_db/')
def check_db():
    logging.info("checking DB")

    sess_dict = get_sess()
    kws = sess_dict['keyword'][0]
    type_ = sess_dict['type_'][0]
    df_file = look_up_in_db(kws, type_)

    if df_file is not None and os.path.exists(df_file):
        logging.info("df file found in DB")
        update_sql('df_file', df_file, 'string')
        html = process_data_in_db(df_file)
        return html

    else:
        df_file = os.path.join(data_dir, 'df_dir', mk_random_string() + '.csv')
        logging.info("df file path: %s" % df_file)
        update_sql('df_file', df_file, 'string')
        logging.info("no df file found in DB, run_analysis")
        return run_analysis()

@app.route("/run_analysis/")
def run_analysis():
    logging.info("starting run_analysis %s" % time.strftime("%H:%M:%S"))
    sess_dict = get_sess()

    ind = indeed_scrape.Indeed(query_type=sess_dict['type_'][0])
    ind.query = sess_dict['keyword'][0]
    ind.stop_words = stop_words
    ind.main()

    index = sess_dict['ind'][0]
    end = sess_dict['end'][0]

    start_time = time.time()
    index, end, num_res, count = ind.get_data(ind=index, start=end)
    logging.debug("end:%i" % end)

    while count < sess_dict['count_thres'][0] and end < num_res:
        index, end, num_res, count = ind.get_data(ind=index, start=end)
        end_time = time.time()
        if (end_time - start_time) / 60.0 > 3.0: # avoid 502
            logging.info("avoiding 502, break")
            break

    update_sql('end', end, 'int')
    update_sql('count_thres', 25, 'int')

    #scrub repeated words
    ind.clean_dup_words()

    # append existing df if second or more time here
    if os.path.exists(sess_dict['df_file'][0]):
        df = load_csv()
        df = df.append(ind.df, ignore_index=True)
    else:
        df = ind.df

    df = ind.summary_similarity(df, 'summary', 80)
    df.dropna(subset=['summary', 'url', 'summary_stem'], how='any', inplace=True)
    df.reset_index(inplace=True, drop=True)
    update_sql('ind', df.shape[0], 'int')

    # save df for additional analysis
    inv = json.dumps(ind.stem_inverse)
    update_sql('stem_inv', inv, 'string')

    save_to_csv(df)

    html = bigram(df, sess_dict['type_'][0], ind)

    update_sql('bigram', html, 'string')

    return html

def bigram(df, type_, ind):
    max_df = compute_max_df(type_, df.shape[0], n_min=2)

    try:
        count_array, kw = ind.vectorizer(df['summary_stem'],
                                        n_min=2,
                                        n_max=3,
                                        max_features=30,
                                        max_df=max_df,
                                        min_df=3)
    except ValueError:
        count_array, kw = ind.vectorizer(df['summary_stem'],
                                        n_min=2,
                                        n_max=3,
                                        max_features=30,
                                        max_df=1.0,
                                        min_df=1)

    kw = get_inverse_stem(kw)
    script, div = get_plot_comp(kw, count_array, df)

    output = """
%(kw_div)s
%(kw_script)s
"""
    logging.debug("run_analysis returning components")
    logging.debug("bokeh div:%s" % div)
    return output %{'kw_script':script,
                    'kw_div':div }

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

    kw = ind.stemmer_(kw)
    words = ind.build_corpus_from_sent(kw, 'summary_stem')
    words = ind.find_words_in_radius(words, kw, 5)

    words = get_inverse_stem(words)

    try:
        count, kw = ind.vectorizer(words, max_features=50, n_min=1, n_max=2,
               max_df=0.90, min_df=2)
    except ValueError:
        return "The body of words compiled did not contain substantially repeated terms."

    script, div = get_plot_comp(kw, count, df)
    return render_template('radius.html', div=div, script=script)

def get_inverse_stem(kw):
    sess_dict = get_sess()
    inv = json.loads(sess_dict['stem_inv'][0])

    orig_keyword = []
    temp = []
    for key in kw: # could be bigrams or more
        keys = key.split(" ")
        for k in keys:
            try: # make sure there's an output
                temp.append(inv[k])
            except KeyError:
                logging.error("inverse stem lookup fail:%s" % k)
                temp.append(k)
        orig_keyword.append(" ".join(temp))
        temp = []

    return orig_keyword

@app.route('/stem/')
def stem():
    logging.info("running stem")
    sess_dict = get_sess()
    df_file = sess_dict['df_file'][0]
    df = load_csv()
    summary_stem = df['summary_stem']

    ind = indeed_scrape.Indeed("kw")
    ind.stop_words = stop_words
    ind.add_stop_words()

    count, kw = ind.vectorizer(summary_stem, n_min=1, n_max=2, max_features=80,
            max_df=compute_max_df(sess_dict['type_'][0], df.shape[0]))

    orig_keywords = get_inverse_stem(kw)
    script, div = get_plot_comp(orig_keywords, count, df)

    page = render_template('stem.html', script=script, div=div)
    return encode_utf8(page)

@app.route('/grammar/')
def grammar_parser():
    logging.info("running grammar parser")
    df = load_csv()
    df.dropna(subset=['grammar'], inplace=True)

    ind = indeed_scrape.Indeed('kw')
    ind.stop_words = stop_words
    ind.add_stop_words()

    count, kw = ind.tfidf_vectorizer(df['grammar'], n_min=2, n_max=3, max_features=30,
            max_df=0.75, min_df=0.01)

    script, div = get_plot_comp(kw, count, df)

    page = render_template('grammar.html', script=script, div=div)
    return encode_utf8(page)

@app.route("/check_count/")
def check_for_low_count_using_title():
    df = load_csv()
    logging.info("title count:%i" % df.shape[0])

    string = "<br><p><i>Your search by job title returned a small number of job \
    postings. Click 'NEW SEARCH' and use the 'keyword' search.</i></p><br>"

    if df.shape[0] < 25:
        return encode_utf8(string)
    else:
        return ""

@app.route('/missing/', methods=['GET', 'POST'])
def compute_missing_keywords():
    if request.method == "POST":
        resume_file = request.files['File']
        logging.info("resume path: %s" % resume_file.filename)

        resume_path = os.path.join(data_dir, resume_file.filename)
        resume_file.save(resume_path)

        df = load_csv()
        rows = missing_keywords.main(resume_path, df['summary'])

        return render_template('missing.html', rows=rows)

    else:
        return render_template('missing.html')

def compute_max_df(type_, num_samp, n_min=1):
    if type_ == 'keywords':
        base = 0.80

    elif type_ == 'title':
        base = 0.85

    else:
        raise ValueError, "type not understood"

    if num_samp > 95:
        base -= 0.10

    elif n_min > 1:
        base -= 0.05

    else:
        pass

    return base

def get_sess():
    sql = "SELECT * FROM data WHERE session_id = '%s';"
    sql = sql % session['session_id']
    sql_engine = sqlalchemy.create_engine(conn_string)

    return pd.read_sql(sql=sql, con=sql_engine)

def update_sql(field, value, data_type):
    sql_engine = sqlalchemy.create_engine(conn_string)

    if data_type == 'string':
        value = escape_string(value)
        sql = "UPDATE data SET `%(field)s` = '%(value)s' WHERE `session_id` = '%(id)s';"

    elif data_type == 'int':
        sql = "UPDATE data SET `%(field)s` = %(value)i WHERE `session_id` = '%(id)s';"

    else:
        raise Exception, "update type not understood"

    sql = sql % {'field':field,
                 'value':value,
                 'id':session['session_id']
                 }

    conn = sql_engine.connect()
    conn.execute(sql)
    conn.close()

def to_sql(sess_dict):
    reference = pd.DataFrame(sess_dict, index=[0])

    sql_engine = sqlalchemy.create_engine(conn_string)
    reference.to_sql(name='data', con=sql_engine, if_exists='append', index=False)

def load_csv():
    sess_dict = get_sess()
    df = pd.read_csv(sess_dict['df_file'][0])

    return df

def save_to_csv(df):
    logging.info("saving df")
    sess_dict = get_sess()
    df.to_csv(sess_dict['df_file'][0], index=False, quoting=1, encoding='utf-8')

def _escape_html(html):
    return html.replace("%", "\%").replace("_", "\_").replace("'", "\'").replace('"', '\"')


if __name__ == "__main__":
    app.run(threaded=True)

