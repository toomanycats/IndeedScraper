import logging
import urllib2
import re
import bs4
import pandas as pd
import indeed_scrape
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import nltk
import pickle
import os
import json
import pdb
import sklearn.linear_model.SGDClassifer

data_dir = os.getenv('OPENSHIFT_DATA_DIR')
if data_dir is None:
    data_dir = os.getenv('PWD')

try: # for calling these methods from CLI
    logging = logging.getLogger(__name__)
except:
    pass

class Resume(object):
    def __init__(self, field, keyword_string):
        self.kw_string = keyword_string
        self.subject = field

    def format_degree(self):
        subject_list = indeed_scrape.Indeed._split_on_spaces(self.subject)
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

            if len(title_comp) != 0:
                for i in [-1, -2]: # take last two job titles
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
        titles, companies = self.filter_titles(titles, companies)

        return titles, companies

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

       stop_words = set((ENGLISH_STOP_WORDS, ('amp', 'and', 'the', 'assistant','intern')))
       vec = CountVectorizer(stop_words=stop_words)
       matrix = vec.fit_transform(df['description'])

       clf = sklearn.linear_model.SGDClassifer()
       clf.fit(matrix, df['titles'])

       f = open(os.path.join(data_dir, 'trained_classifier.pickle'))
       pickle.dump(clf, f)
       f.close()

       f = open(os.path.join(data_dir, 'trained_vectorizer.pickle'))
       pickle.dump(vec, f)
       f.close()









