######################################
# File Name : indeed_scrape.py
# Author : Daniel Cuneo
# Creation Date : 11-05-2015
######################################
#import GrammarParser
import codecs
import re
import ConfigParser
import logging
import json
import pandas as pd
import urllib2
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import numpy as np
from nltk import stem
from nltk import tokenize
import re
import os
import pdb

#grammar = GrammarParser.GrammarParser()

toker = tokenize.word_tokenize
stemmer = stem.SnowballStemmer('english')

repo_dir = os.getenv("OPENSHIFT_REPO_DIR")
if repo_dir is None:
    repo_dir = os.getenv("PWD")

logging = logging.getLogger(__name__)

class Indeed(object):
    def __init__(self, query_type):
        self.query_type = query_type
        self.delta_zero = 0
        self.zip_code_error_limit = 300
        self.num_urls = 10
        self.add_loc = None
        self.stop_words = None
        self.num_samp = 1000
        self.zip_code_file = os.path.join(repo_dir, 'us_postal_codes.csv')
        self.df = pd.DataFrame(columns=['url', 'job_key', 'summary', 'summary_stem', 'city', 'zipcode', 'jobtitle'])
        self.config_path = os.path.join(repo_dir, "tokens.cfg")
        self.query = None
        self.title = None
        self.locations = None
        self.radius = 1

    def _decode(self, string):
        try:
            string = string.decode("utf-8", "ignore").encode("utf-8", "ignore")
            return string

        except Exception:
            return string

    @classmethod
    def _split_on_spaces(self, string):
        ob = re.compile('\s+')
        return ob.split(string.strip())

    def add_stop_words(self):
        if self.stop_words is not None:
            words = self._split_on_spaces(self.stop_words)
            self.stop_words = ENGLISH_STOP_WORDS.union(words)

    def build_api_string(self):
        if self.query is None:
            print "query cannot be empty"
            raise ValueError

        # beware of escaped %
        prefix = 'http://api.indeed.com/ads/apisearch?'
        pub = 'publisher=%(pub_id)s'
        chan = '&chnl=%(channel_name)s'
        loc = '&l=%(loc)s'
        query = self.format_query()
        start = '&start=0'
        frm = '&fromage=600'
        limit = '&limit=100'
        site = '&st=jobsite'
        format = '&format=json'
        filter = '&filter=1'
        country = '&co=us'
        radius = '&radius=%(radius)s'
        suffix = '&userip=1.2.3.4&useragent=Mozilla/%%2F4.0%%28Firefox%%29&v=2'

        self.api = prefix + pub + chan + loc + query + start + frm + limit + \
                   site + format + country + radius + filter + suffix

        logging.debug("api string: %s" % self.api)

    def format_keywords(self, string):
        kws = "+".join(self._split_on_spaces(string))
        return kws

    def format_query(self):
        logging.debug("query: %s" % self.query)

        if self.query_type == 'title':
            return '&q=title%%3A%%28%(query)s%%29'

        elif self.query_type == 'keywords':
            return '&q=%(query)s'

        elif self.query_type == 'keywords_title':
            raise NotImplementedError, "not implemented yet"
            #if self.title is None:
            #    raise ValueError, "no title set"
            #return '&q=%(kws)s+title%%3A%%28%(title)s%%29'

        else:
            raise Exception, "not a recogized type"

    def load_config(self):
        '''loads a config file that contains tokens'''
        config_parser = ConfigParser.RawConfigParser()
        config_parser.read(self.config_path)
        self.pub_id = config_parser.get("id", "pub_id")
        self.channel_name = config_parser.get("channel", "channel_name")

    def load_zipcodes(self):
        '''loads the zip code file and returnes a list of zip codes'''
        self.df_zip = pd.read_csv(self.zip_code_file, dtype=str)
        self.df_zip.dropna(inplace=True, how='all')

    def end_url_loop(self, zip_ind):
        df = self.df.dropna(subset=['summary']).drop_duplicates(subset=['job_key'])
        count = df.url.count()
        logging.debug("df count: %i" % count)

        if count  >= self.num_urls:
            logging.info("ending reached count")
            return True, count

        else:
            return False, count

    def count_dupes(self, count, prev_count, num_dupes):
        delta_cnt = count - prev_count

        if  delta_cnt == 0:
            num_dupes += 1

        return num_dupes

    def get_city_url_content_stem(self):
        ind = 0
        prev_count = 0
        count = 0
        num_dupes = 0

        for zip_ind, zipcode in enumerate(self.locations):
            logging.debug("zip ind: %i" % zip_ind)

            if count == 0 and zip_ind == self.zip_code_error_limit:
                error_string = "Your query isnt' matching."
                raise Exception, error_string

            prev_count = count

            # periodic check
            if np.mod(zip_ind, 10) == 0:
                end_bool, count = self.end_url_loop(zip_ind)
                if end_bool:
                    self.df = self.df.dropna(subset=['summary'])
                    # cheap insurance
                    self.df.drop_duplicates(subset=['job_key'], inplace=True)
                    return
                else:
                    num_dupes = self.count_dupes(count, prev_count, num_dupes)
                    if num_dupes > 100:
                        logging.info("num dupes reached:%i" % num_dupes)
                        return

            url_city_title = self.get_url(zipcode)
            if url_city_title is None:
                logging.debug("url: None found")
                continue

            for item in url_city_title:
                if item[4] == 0:
                    continue
                # avoid dupes
                if self.df['job_key'].isin([item[3]]).any():
                    logging.debug("duplicate job key:%s" % item[3])
                    continue
                try:
                    content, full_text = self.parse_content(item[0])
                    self.df.loc[ind, 'url'] = item[0]
                    self.df.loc[ind, 'city'] = item[1]
                    self.df.loc[ind, 'jobtitle'] = item[2]
                    self.df.loc[ind, 'job_key'] = item[3]
                    self.df.loc[ind, 'summary'] = content
                    self.df.loc[ind, 'summary_stem'] = self.stemmer_(content)
                    #self.df.loc[ind, 'full_text'] = grammar.main(full_text)
                    ind += 1

                    logging.debug("index: %i" % ind)
                    logging.debug("count: %i" % count)
                    logging.debug("index increase: %i" % ind)

                except:
                    pass


    def get_url(self, location):

        api = self.api %{'pub_id':self.pub_id,
                         'loc':location,
                         'channel_name':self.channel_name,
                         'query':self.format_keywords(self.query),
                         'radius':self.radius
                        }

        logging.debug("full api:%s" % api)

        try:
            response = urllib2.urlopen(api)
            data = json.loads(response.read())
            response.close()

            urls = []
            urls.extend([(item['url'], item['city'], item['jobtitle'], item['jobkey'], data['totalResults'])  for item in data['results']])

        except urllib2.HTTPError, err:
            logging.debug("get url: %s" % err)
            return None

        except Exception, err:
            logging.debug("get url: %s" % err)
            return None

        return urls

    def get_content(self, url):
        if url is None:
            return None

        try:
            response = urllib2.urlopen(url)
            content = response.read()
            response.close()

            return content

        except urllib2.HTTPError, err:
            logging.error("get content:%s" % err)
            return None

        except Exception, err:
            logging.error("get content:%s" % err)
            return None

    def len_tester(self, word_list):
        new_list = []
        for word in word_list:
            if len(word) < 3:
                continue
            else:
                new_list.append(word)

        return new_list

    def stemmer_(self, string):

        if string is None:
            return None

        string = self._decode(string)
        words = toker(string)
        words = self.len_tester(words)
        words = map(stemmer.stem, words)

        return " ".join(words)

    def parse_content(self, url):
        try:
            content = self.get_content(url)

            if content is None:
                return

            content = self._decode(content)
            soup = BeautifulSoup(content, 'html.parser')

            summary = soup.find('span', {'summary'})
            skills = summary.find_all("li")

            output = [item.get_text() for item in skills]

            if len(output) > 0:
                parsed = " ".join(output).replace('\n', '')

            else:
                # has a summary class but no li
                logging.debug("soup didn't parse summary li:%s" % url)
                output = summary.get_text()
                output = [item for item in output.split("\n")]

                if len(output) > 0:
                    parsed = " ".join(output).replace("\n", "")

                else:
                    output = soup.find_all("li")
                    output = [item.get_text() for item in output]
                    if len(output) > 0:
                        parsed = " ".join(output).replace("\n", "")
                    else:
                        logging.debug("soup didn't parse anything")
                        return None, None

            return parsed, summary.get_text()

        except Exception, err:
            logging.error(err)
            return None

    def parse_zipcode_beg(self, regex):
        '''locs are zipcode prefixes, like:902, provided as string'''
        pat = '^%s' % regex
        obj = re.compile(pat)

        self.df_zip['include'] = self.df_zip['Postal Code'].apply(lambda x: 1 if obj.match(x) else 0)
        zips = self.df_zip[self.df_zip['include']==1]['Postal Code']

        return zips.tolist()

    def handle_locations(self):
        '''main method for setting up the locations for the API call'''
        locations = []
        self.load_zipcodes()
        for regex in self.add_loc: #list of regex
            for loc in self.parse_zipcode_beg(regex):
                locations.append(loc)

        other_loc = self.df_zip['Postal Code'].sample(self.num_samp).tolist()
        locations.extend(other_loc)
        locations = np.unique(locations)
        np.random.shuffle(locations)

        return locations

    def save_data(self):

        self.df.drop_duplicates(subset=['url'], inplace=True)
        self.df.dropna(inplace=True, how='any', axis=0)

        self.df.to_csv('/home/daniel/git/Python2.7/DataScience/indeed/data_frame.csv',
                        index=False,
                        encoding='utf-8')

    def main(self):
        '''Run all the steps to collect data and produce a bi-gram
        bar plot of the keyword counts. Terminating the program execution
        with control C, will save whatever data was collected. To make this
        save-on-quit feature more usable, the locations are shuffled prior to
        getting the content.'''

        self.load_config()
        self.build_api_string()
        self.add_stop_words()
        self.locations = self.handle_locations()

        self.get_city_url_content_stem()

    def vectorizer(self, corpus, max_features=200, max_df=0.8, min_df=0.1, n_min=2, n_max=3):
        vectorizer = CountVectorizer(max_features=max_features,
                                    max_df=max_df,
                                    min_df=min_df,
                                    lowercase=True,
                                    stop_words=self.stop_words,
                                    ngram_range=(n_min, n_max),
                                    analyzer='word',
                                    decode_error='ignore',
                                    strip_accents='unicode'
                                    )

        matrix = vectorizer.fit_transform(corpus)
        features = vectorizer.get_feature_names()

        return matrix, features

    def find_words_in_radius(self, series, keyword, radius):

        words_in_radius = []

        for string in series:
            try:
                test = string.decode("utf-8", "ignore")
            except:
                continue
            test = tokenize.word_tokenize(test)
            test = self.len_tester(test)
            test = np.array(test)

            kw_ind = np.where(test == keyword)[0]
            if len(kw_ind) == 0: #empty
                continue

            src_range = self._find_words_in_radius(kw_ind, test, radius)

            temp = " ".join(test[src_range])
            words_in_radius.append(temp)

        return words_in_radius

    def _find_words_in_radius(self, kw_ind, test, radius):
            # can be more than one kw in a string
            lower = kw_ind - radius
            upper = kw_ind + radius

            src_range_tot = np.empty(0, dtype=np.int16)

            for i in range(lower.shape[0]):
                src_range = np.arange(lower[i], upper[i])

                # truncate search range to valid values
                # this operation also flattens the array
                src_range = src_range[(src_range >= 0) & (src_range < test.size)]
                src_range_tot = np.hstack((src_range_tot, src_range))

            return np.unique(src_range_tot)

    def build_corpus_from_sent(self, keyword):
        #pdb.set_trace()
        documents = self.df['summary'].apply(lambda x: x.decode("utf-8", "ignore"))
        corpus = []
        for doc in documents:
            corpus.extend(tokenize.sent_tokenize(doc))

        new = []
        obj = re.compile(keyword, re.I)

        for sentence in corpus:
            if obj.search(sentence):
                new.append(sentence)

        return new


if __name__ == "__main__":
    ind = Indeed()
    ind.main()
