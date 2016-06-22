import logging
import urllib2
import re
import bs4
import pandas as pd
import indeed_scrape
import numpy as np
import os
import json
import pdb
from fuzzywuzzy import fuzz

data_dir = os.getenv('OPENSHIFT_DATA_DIR')
if data_dir is None:
    data_dir = os.getenv('PWD')

try: # for calling these methods from CLI
    logging = logging.getLogger(__name__)
except:
    pass

class Resume(object):
    def __init__(self, field=None, title=None, location=None, keyword_string=None):
        self.kw_string = keyword_string
        self.subject = field
        self.title = title
        self.loc = location
        self.api_base = "http://www.indeed.com/resumes"

    def format_degree(self):
        subject_list = indeed_scrape.Indeed._split_on_spaces(self.subject)
        formatted_subject = "+".join(subject_list)
        logging.debug("formatted degree subject:%s" % formatted_subject)

        return formatted_subject

    def format_kw_query(self):

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

    def format_location_string(self):
        loc_string = ''
        loc_parts = indeed_scrape.Indeed._split_on_spaces(self.loc)
        loc_string += "+".join(loc_parts)

        return loc_string

    def get_title_string(self):
        string = ''
        title_parts = indeed_scrape.Indeed._split_on_spaces(self.title)
        string += "+".join(title_parts)

        return string

    def get_title_api(self, page=0):
        title_string = self.get_title_string()
        location_string = self.format_location_string()

        title = '?q=%(title_string)s'
        title = title % {'title_string':title_string}
        #suffix = '&co=US&rb=yoe%%3A12-24'
        where = '&l=%s' % location_string
        pagination = '&start=%i' % page

        api = self.api_base + title + where + pagination
        logging.debug("api:%s" % api)

        return api

    def get_kw_api(self, page=0):
        formatted_query = self.format_query()
        logging.debug("formatted query:%s" % formatted_query)
        if formatted_query is None:
            keywords = '?q='
        else:
            keywords = '?q=%s+' % formatted_query

        formated_subject = self.format_degree()

        degree = 'fieldofstudy%%3A%(field)s' % {'field':formated_subject}
        suffix = '&co=US&rb=yoe%%3A12-24'
        pagination = '&start=%i' % page

        api = self.api_base + keywords + degree + suffix + pagination
        logging.debug("api:%s" % api)

        return api

    def get_html_from_api(self, api):
        try:
            response = urllib2.urlopen(api)
            html = response.read()
            response.close()

            return html

        except Exception, err:
            logging.error(err)

            return None

    def get_full_resume_links(self, html):
        soup = bs4.BeautifulSoup(html, 'lxml')
        divs = soup.find_all('div', {'class': 'clickable_resume_card'})
        divs = map(str, divs)

        pat = '\<div class=\"clickable_resume_card\" onclick=\"window\.open\(.*?\)'
        obj = re.compile(pat)

        links_noisy = map(lambda x: obj.search(x).group(), divs)

        path_pat = '\/r\/.*?(\')'
        clean_links = []

        for l in links_noisy:
            match = re.search(path_pat, l)
            if match:
                ll = match.group().replace("'", "")
                clean_links.append(ll)

        return clean_links

    def get_des_from_res(self, link):
        try:
            base = "http://www.indeed.com"

            resp = urllib2.urlopen(base + link)
            html = resp.read()
            resp.close()

            soup = bs4.BeautifulSoup(html, 'lxml')
            des_list = soup.find_all("p", {"class":"work_description"})

            pat = '\<p class\=\"work_description\"\>(?P<des>.*?)(\>)'
            ob = re.compile(pat)

            descriptions = []
            for des in des_list:
                des = str(des)
                match = ob.search(des)
                if match:
                    descriptions.append(match.group("des"))

            des = self._clean_res_des(descriptions)

            return des

        except Exception, err:
            print err
            logging.error(err)
            return None

    def _clean_res_des(self, des_list):
        return map(lambda x: x.decode("ascii", "ignore").replace("&amp;", "and").replace("<br/", ""), des_list)

    def parse_html(self, html):
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

            if len(title_comp) > 1:
                results.extend(title_comp)

        return results

    def _clean_target(self, string):
        return string.replace("<", "").replace("...", "")

    def remove_universities(self, companies, titles):
        obj = re.compile("(?i)university")
        obj_grad = re.compile("(?i)undergraduate|undergrad|graduate|grad|postdoc|college")
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
        titles = map(lambda x: re.sub("^\s+", "", x), titles)
        titles = map(lambda x: re.sub("$\s+", "", x), titles)
        titles = map(lambda x: re.sub("\s{2,}", " ", x), titles)

        out = []
        for title in titles:
            #temp_list = ind._decode(title)
            #if temp_list is None:
                #continue
            temp_list = indeed_scrape.toker(title)
            temp_list = ind.len_tester(temp_list)
            temp_string = " ".join(temp_list)
            temp_string = re.sub('\s+amp\s+', ' ', temp_string)
            temp_string = re.sub('\s+and\s+', ' ', temp_string)
            out.append(temp_string)

        out = map(lambda x: x.lower(), out)

        return out

    def filter_hier_titles(self, titles, companies):
        pattern = "sr\.|senior|jr\.|junior|vice|director|president|visiting|lead"
        pattern += "associate|assistant|"
        pattern += "(?i)" # must always be last

        titles = map(lambda x: re.sub(pattern, '', x), titles)

        temp_titles = []
        temp_companies = []

        for title, comp in zip(titles, companies):
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
        #TODO: logic for diff types of api
        api = self.get_title_api(page)
        html = self.get_html_from_api(api)

        if html is None:
            return None, None

        data = self.parse_html(html)

        titles = map(lambda x: x[0], data)
        companies = map(lambda x: x[1], data)

        return titles, companies

    def run_loop(self):
        api = self.get_title_api(page=0)
        html = self.get_html_from_api(api)

        num = self.get_number_of_resumes_found(html)
        if num > 5000:
            num = 5000

        titles = []
        companies = []

        for page in  np.arange(0, num, 50):
            temp_titles, temp_companies = self.get_final_results(page)

            if temp_titles is None:
                continue

            titles.extend(temp_titles)
            companies.extend(temp_companies)

        titles = self.normalize_titles(titles)
        titles, companies = self.filter_hier_titles(titles, companies)

        return titles, companies

    def group(self, df, key="title", count="count"):
        cnt = df.groupby(key).count()
        cnt.dropna(how='any', inplace=True)
        cnt.sort(count, inplace=True)

        return cnt

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

    def ratio_norm_titles(self, labels, df, column, ratio_thres):
        """Use on grouped df"""

        n = len(labels)
        for i in range(n):
            try:
                string1 = labels[i]
                best_r = 0.0

                for j in range(df.shape[0]-1):
                    string2  = df.loc[j, column]
                    ratio = fuzz.ratio(string1, string2)

                    if ratio < 100.0 and ratio >= ratio_thres and ratio > best_r:
                        r_best = ratio
                        print "orig:%s new:%s" %(string2, labels[i])
                        df.loc[j, column] = labels[i]

            except Exception, err:
                print err
                logging.error("summary similarity error: %s" % err)
                continue

        return df

    def inverse_stem_titles(self, titles, inv_stem):
        new_titles = []
        for t in titles:
            words = indeed_scrape.Indeed._split_on_spaces(t)
            temp = map(lambda x: inv_stem[x], words)
            string = " ".join(temp)
            new_titles.append(string)

        return new_titles

    def main(self):
        titles, comps = self.run_loop()

        ind = indeed_scrape.Indeed(None)
        titles = map(ind.stemmer_, titles)
        ind.stem_inverse[''] = "None"
        self.inv_title_dict = ind.stem_inverse

        df = pd.DataFrame({"title":titles,
                           "comp":comps,
                           "count":np.ones(len(titles))},
                           index=np.arange(len(titles))
                        )

        df.drop_duplicates(subset=['comp', 'title'], inplace=True)
        df.reset_index(inplace=True)

        ### final steps ###
        labels = list(self.group(df)[-30:].index)
        test = self.ratio_norm_titles(labels, df, 'title', 90)
        out = self.group(test)
        inv_titles = self.inverse_stem_titles(out.index, self.inv_title_dict)
        out['inv_title'] = inv_titles

        return df, out

    def plot(self, df, x='inv_title', y='count'):
        df.plot(kind='bar', x=x, y=y, rot=90, fontsize="large", grid=True)

        plt.title("Titles From Resume Search Results:'Product Manager'", fontsize="large")
        plt.xlabel("Titles", fontsize="large")
        plt.ylabel("Counts", fontsize="large")

    def add_perc_col_to_cnt(self, df):

        df = df.copy()
        df['perc'] = 100.0 * df['count'] / df['count'].sum()

        return df
