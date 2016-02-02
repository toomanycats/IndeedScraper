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
import pdb

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
                for i in  [-1]:
                    try:
                        results.append(title_comp[i])
                    except IndexError:
                        pass

        return results

    def _clean_target(self, string):
        return string.replace("<", "").replace("...", "")

    def remove_universities(self, companies, titles):
        obj = re.compile("(?i).*universit.*")
        remaining = []

        for tit, com in zip(titles, companies):
            match = obj.search(com)
            if not match:
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

    def filter_titles(self, titles):

        titles = map(lambda x: x.replace("sr.", ""), titles)
        titles = map(lambda x: x.replace("senior", ""), titles)
        titles = map(lambda x: x.replace("jr.", ""), titles)
        titles = map(lambda x: x.replace("junior", ""), titles)

        titles = map(lambda x: re.sub('^\s+', '', x), titles)

        temp = []
        for title in titles:
            if titles == '':
                continue
            else:
                temp.append(title)

        titles = temp

        return titles

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

    def count_words_in_titles(self, titles):
        counter = {}
        bag = []

        for t in titles:
            bag.extend(indeed_scrape.toker(t))

        for word in bag:
            if not counter.has_key(word):
                counter[word] = 1
            else:
                counter[word] += 1

        return sorted(counter.items(), key=lambda x: x[1], reverse=True)

    def match_words_to_titles(self, words, titles):
        return_titles = []
        for title in titles:
            for word in words:
                if word in title:
                    return_titles.append(title)

        return return_titles

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
        for page in  np.arange(0, num, 50):
            temp_titles, _ = self.get_final_results(page)
            titles.extend(temp_titles)

        titles = self.normalize_titles(titles)
        titles = self.filter_titles(titles)

        return titles

    def top_words(self, df):
        out = []
        for i in range(20):
            temp = df[df['group_index'] == i]['titles']
            out.append(self.count_words_in_titles(temp)[0])

        out.sort(key=lambda x:x[1], reverse=True)

        return out

    def kmeans(self, titles):
        ind = indeed_scrape.Indeed("kw")
        ind.stop_words = "and amp"
        ind.add_stop_words()
        matrix, features = ind.vectorizer(titles,
                                max_features=300,
                                max_df=1.0,
                                min_df=3,
                                n_min=1,
                                n_max=1
                                )

        #km = DBSCAN(eps=0.1, min_samples=2)
        km = KMeans(20)
        assignments = km.fit_predict(matrix)

        df = pd.DataFrame({'titles':titles,
                           'group_index':assignments
                           }
                         )

        return df, features


