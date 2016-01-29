import urllib2
import re
import bs4
import pandas as pd
import indeed_scrape
import pandas as pd
import numpy as np
import pdb

class Resume(object):
    def __init__(self, field, keyword_string):
        self.kw_string = keyword_string
        self.subject = field

    def format_query(self):

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
        base = "http://www.indeed.com/resumes"
        keywords = '?q=%s+' % formatted_query
        degree = 'fieldofstudy%%3A%(field)s' % {'field':self.subject}
        suffix = '&co=US&rb=yoe%%3A12-24'
        pagination = '&start=%i' % page

        api = base + keywords + degree + suffix + pagination
        return api

    def get_html_from_api(self, api):
            response = urllib2.urlopen(api)
            html = response.read()
            response.close()

            return html

    def parse_data(self, html):
        obj = re.compile('\s\-\s(?P<target>\w+.*?\<)')

        soup = bs4.BeautifulSoup(html)
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
                results.append(title_comp[-1])

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

    def get_final_results(self, page=0):
        api = self.get_api(page)
        html = self.get_html_from_api(api)
        num = self.get_number_of_resumes_found(html)
        data = self.parse_data(html)

        titles = map(lambda x: x[0], data)
        companies = map(lambda x: x[1], data)

        titles = self.normalize_titles(titles)
        titles = self.filter_titles(titles)

        titles, companies = self.sort_results(titles, companies)

        #groups = self.group_companies(data)
        #return groups
        return titles, companies, num

    def normalize_titles(self, titles):
        ind = indeed_scrape.Indeed("kw")

        titles = map(lambda x: x.replace("/", " "), titles)
        titles = map(lambda x: x.replace("-", " "), titles)

        out = []
        for title in titles:
            temp_list = indeed_scrape.toker(title)
            temp_list = ind.len_tester(temp_list)
            temp_string = " ".join(temp_list[0:3])
            out.append(temp_string)

        out = map(lambda x: x.lower(), out)

        return out

    def filter_titles(self, titles):

        titles = map(lambda x: x.replace("sr.", ""), titles)
        titles = map(lambda x: x.replace("senior", ""), titles)
        titles = map(lambda x: x.replace("jr.", ""), titles)
        titles = map(lambda x: x.replace("junior", ""), titles)

        titles = map(lambda x: re.sub('^\s+', '', x), titles)

        return titles

    def sort_results(self, titles, companies):
        titles = np.array(titles)
        companies = np.array(companies)

        ind_sort = np.argsort(titles)

        titles = titles[ind_sort]
        companies = companies[ind_sort]

        return titles, companies

    def get_number_of_resumes_found(self, html):
        soup = bs4.BeautifulSoup(html)
        div = soup.find("div", {'id':'result_count'})
        count_string = re.search('\>\s*(?P<num>.*?\<)', str(div)).group("num")
        count_string = count_string.replace("<", "")
        count_string = count_string.replace(",", "")
        count_string = count_string.replace("resumes", "",)
        count_string = count_string.replace(" ", "",)

        count = int(count_string)

        return count














