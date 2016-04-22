import logging
import urllib2
import re
import bs4
import pandas as pd
import indeed_scrape
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn import cross_validation
from sklearn import metrics
import nltk
import pickle
import os
import json
import pdb
import sklearn
import nltk

data_dir = os.getenv('OPENSHIFT_DATA_DIR')
if data_dir is None:
    data_dir = os.getenv('PWD')

try: # for calling these methods from CLI
    logging = logging.getLogger(__name__)
except:
    pass

class Resume(object):
    def __init__(self, job_title=None, keyword_string=None):
        self.kw_string = keyword_string
        self.title= job_title

    def format_degree(self):
        subject_list = indeed_scrape.Indeed._split_on_spaces(self.title)
        formatted_subject = "+".join(subject_list)
        logging.debug("formatted degree subject:%s" % formatted_subject)

        return formatted_subject

    def format_query(self):

        if self.kw_string is None:
            return None

        query_parts = indeed_scrape.Indeed._split_on_spaces(self.kw_string)

        if len(query_parts) == 1:
            format = "%s" % query_parts[0]

        elif len(query_parts) == 2:
            format = "(%s-OR-%s)" % (query_parts[0], query_parts[1])

        elif len(query_parts) > 2:
            format = "("
            format += "-OR-".join(query_parts)
            format += ")"

        else:
            raise ValueError, "Length of query string must be greater than zero"

        return format

    def get_api(self, page=0):
        formatted_query = self.format_query()
        logging.debug("formatted query:%s" % formatted_query)
        if formatted_query is None:
            keywords = '?q='
        else:
            keywords = '?q=%s+' % formatted_query

        formated_subject = self.format_degree()

        base = "http://www.indeed.com/resumes"
        degree = 'fieldofstudy%%3A%(field)s' % {'field':formated_subject}
        suffix = '&co=US&rb=yoe%%3A12-24'
        pagination = '&start=%i' % page

        api = base + keywords + degree + suffix + pagination
        logging.debug("api:%s" % api)
        return api

    def get_html_from_api(self, api):
            response = urllib2.urlopen(api)
            html = response.read()
            response.close()

            return html

    def parse_data(self, html):
        obj = re.compile('\s\-\s(?P<target>\w+.*?\<)')

        soup = bs4.BeautifulSoup(html, "lxml")
        persons = soup.find_all('li')

        results = []
        for pers in persons:
            try:
                titles = re.findall('class="experience"\>(?P<exp>.*?\<)', str(pers))
                titles = map(self._clean_target, titles)

                companies = re.findall('class="company"\>(?P<comp>.*?\<)', str(pers))
                companies = map(lambda x: obj.match(x).group("target"), companies)
                companies = map(self._clean_target, companies)
            except:
                continue

            title_comp = self.remove_universities(companies, titles)

            #TODO: improve this
            if len(title_comp) != 0:
                for i in [-1, -2, -3]: # take last three job titles
                    try:
                        results.append(title_comp[i])
                    except IndexError:
                        pass

        return results

    def _clean_target(self, string):
        return string.replace("<", "").replace("...", "")

    def remove_universities(self, companies, titles):
        obj = re.compile("(?i)university")
        obj_grad = re.compile("(?i)undergraduate|undergrad|graduate|grad|postdoc")
        remaining = []

        for tit, com in zip(titles, companies):
            match_com = obj.search(com)
            match_tit = obj.search(tit)
            match_grad= obj_grad.search(tit)

            if match_com or match_tit or match_grad:
               continue

            else:
                remaining.append((tit, com))

        return remaining

    def group_companies(self, companies):
        df = pd.DataFrame({'company_name':companies})
        df['count'] = df['company_name'].apply(lambda x: 1)
        grp = df.groupby("company_name")

        count = grp.count()
        comp = grp.count().index

        return comp, count

    def normalize_titles(self, titles):
        ind = indeed_scrape.Indeed("kw")

        titles = map(lambda x: x.replace("/", " "), titles)
        titles = map(lambda x: x.replace("-", " "), titles)

        out = []
        for title in titles:
            temp_list = ind._decode(title)
            if temp_list is None:
                continue
            temp_list = indeed_scrape.toker(title)
            temp_list = ind.len_tester(temp_list)
            temp_string = " ".join(temp_list)
            out.append(temp_string)

        out = map(lambda x: x.lower(), out)

        return out

    def filter_titles(self, titles, companies):
        obj = re.compile("(?i)volunteer|intern")

        titles = map(lambda x: x.replace("sr.", ""), titles)
        titles = map(lambda x: x.replace("senior", ""), titles)
        titles = map(lambda x: x.replace("jr.", ""), titles)
        titles = map(lambda x: x.replace("junior", ""), titles)
        titles = map(lambda x: x.replace("vice", ""), titles)
        titles = map(lambda x: x.replace("director", ""), titles)
        titles = map(lambda x: x.replace("president", ""), titles)
        titles = map(lambda x: x.replace("visiting", ""), titles)

        titles = map(lambda x: re.sub('^\s+', '', x), titles)

        temp_titles = []
        temp_companies = []
        for title, comp in zip(titles, companies):
            if titles == '' or obj.search(title):
                continue
            else:
                temp_titles.append(title)
                temp_companies.append(comp)

        return temp_titles, temp_companies

    def sort_results(self, titles, companies):
        titles = np.array(titles)
        companies = np.array(companies)

        ind_sort = np.argsort(titles)

        titles = titles[ind_sort]
        companies = companies[ind_sort]

        return titles, companies

    def get_number_of_resumes_found(self, html):
        soup = bs4.BeautifulSoup(html, 'lxml')
        div = soup.find("div", {'id':'result_count'})

        count_string = re.search('\>\s*(?P<num>.*?\<)', str(div)).group("num")
        count_string = count_string.replace("<", "")
        count_string = count_string.replace(",", "")
        count_string = count_string.replace("resumes", "",)
        count_string = count_string.replace(" ", "",)
        logging.info("number of resumes found:%s" % count_string)

        try:
            count = int(count_string)
        except ValueError, err:
            logging.error(err)
            print "No pages found"
            raise ValueError

        return count

    def get_final_results(self, page=0):
        api = self.get_api(page)
        html = self.get_html_from_api(api)
        data = self.parse_data(html)

        titles = map(lambda x: x[0], data)
        companies = map(lambda x: x[1], data)

        titles, companies = self.sort_results(titles, companies)

        #groups = self.group_companies(data)
        #return groups
        return titles, companies

    def run_loop(self):
        api = self.get_api(page=0)
        html = self.get_html_from_api(api)
        num = self.get_number_of_resumes_found(html)
        if num > 1000:
            num = 1000

        titles = []
        companies = []
        for page in  np.arange(0, num, 50):
            temp_titles, temp_companies = self.get_final_results(page)
            titles.extend(temp_titles)
            companies.extend(temp_companies)

        titles = self.normalize_titles(titles)
        #titles, companies = self.filter_titles(titles, companies)

        return titles, companies

    def prepare_plot(self, data):
        df = pd.DataFrame({"data":data,
                           "count": np.ones(len(data))
                          })

        cnt = df.groupby("data").count()
        thres = np.floor(cnt.mean() + cnt.std())
        cnt = cnt[cnt >= thres]
        cnt.dropna(how='any', inplace=True)

        return cnt

    def top_words(self, df):
        out = []
        for i in range(20):
            temp = df[df['group_index'] == i]['titles']
            out.append(self.count_words_in_titles(temp)[0])

        out.sort(key=lambda x:x[1], reverse=True)

        return out

    def categorize_job_titles(self, titles):
        titles = map(lambda x:x.lower(), titles)

        f = open(os.path.join(data_dir, 'trained_classifier.pickle'))
        clf = pickle.load(f)
        f.close()

        g = open(os.path.join(data_dir, "trained_vectorizer.pickle"))
        vec = pickle.load(g)
        g.close()

        matrix = vec.transform(titles)

        out = {}
        for ind, title in enumerate(titles):
            row = matrix[ind, :]
            label = clf.predict(row)
            out[title] = label[0].decode("ascii", "ignore")

        return out


class Train(object):
    def __init__(self, master_file):
        self.master_file = master_file

    def main(self):
       dict_ = json.load(open(self.master_file))
       df = pd.DataFrame({"titles":dict_.keys(),
                          "description":dict_.values()}
                          )

       df['description'] = df['description'].apply(" ".join)

       #helper = Helper()
       #df = helper.stem_df(df, "description")

       stop_words = set((ENGLISH_STOP_WORDS, ('amp', 'and', 'the', 'intern')))

       title_clf = Pipeline([
                            ('vec', CountVectorizer(stop_words=stop_words,
                                                    binary=True,# hack uniq,
                                                    ngram_range=(1, 1),
                                                    decode_error="ignore")),
                            ('clf', SGDClassifier(loss='log',
                                                  alpha=1e-4,
                                                  shuffle=False,
                                                  penalty='l1',
                                                  random_state=0))
                            ])


       title_clf.fit(df['description'], df['titles'])
       f = open(os.path.join(data_dir, 'trained_classifier.pickle'), 'wb')
       pickle.dump(title_clf, f)
       f.close()

       return title_clf

    def get_df_for_cv(self):
        #helper = Helper()
        dict_ = json.load(open(self.master_file))
        df = pd.DataFrame()

        # for cross validation, we need repeated labels
        #df['description'] = df['description'].apply(" ".join)
        index = 0
        for key, values in dict_.iteritems():
            for v in values:
                df.loc[index, 'description'] = v
                df.loc[index, 'title'] = key
                index += 1

        return df

    def grid_search(self):
        df = self.get_df_for_cv()

        stop_words = set((ENGLISH_STOP_WORDS, ('amp', 'and', 'the' )))

        title_clf = Pipeline([
                             ('vec', CountVectorizer(stop_words=stop_words,
                                                     binary=True,# hack uniq,
                                                     ngram_range=(1, 1),
                                                     decode_error="ignore")),
                             ('clf', SGDClassifier(random_state=0))
                             ])

        labels = df['title'].unique()
        cv = cross_validation.KFold(df.shape[0], 10)
        params = {"vec__ngram_range":[(1,1), (1,2)],
                  "clf__alpha":(1e-1, 1e-2, 1e-3, 1e-4),
                  "clf__loss":['hinge', 'squared_hinge', 'log', 'squared_loss'],
                  "clf__penalty":('l1', 'l2'),
                  "clf__shuffle":[True, False]
                  }

        gs = GridSearchCV(title_clf, params, cv=cv, n_jobs=6)
        gs.fit(df['description'], df['title'])

        best_parameters, score, _ = max(gs.grid_scores_, key=lambda x: x[1])
        for param_name in sorted(params.keys()):
            print("%s: %r" % (param_name, best_parameters[param_name]))


class Helper(object):
    def __init__(self):
        self.stemmer = nltk.stem.LancasterStemmer()

    def test_ending(self, word):
        if len(word) >= 4 and  word[-1] == 's':
            word = self.stemmer.stem(word)

        elif len(word) >= 5 and word[-2] == "er":
            word  = self.stemmer.stem(word)

        elif len(word) >= 6 and word[-3] == 'ing':
            word = self.stemmer.stem(word)

        else:
           pass

        return word

    def string_stemmer(self, string):
        words = indeed_scrape.toker(string)
        out = []
        for word in words:
            word = self.test_ending(word)
            out.append(word)

        return " ".join(out)

    def stem_df(self, df, key):
        df[key] = df[key].apply(self.string_stemmer)
        return df

