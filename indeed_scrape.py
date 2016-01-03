#####################################
# File Name : indeed_scrape.py
# Author : Daniel Cuneo
# Creation Date : 11-05-2015
######################################
import pdb
from fuzzywuzzy import fuzz
import GrammarParser
import codecs
import re
import ConfigParser
import logging
import json
import pandas as pd
import urllib2
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
import numpy as np
from nltk import stem
from nltk import tokenize
import re
import os

grammar = GrammarParser.GrammarParser()

toker = tokenize.word_tokenize
stemmer = stem.SnowballStemmer('english')

repo_dir = os.getenv("OPENSHIFT_REPO_DIR")
if repo_dir is None:
    repo_dir = os.getenv("PWD")

try: # for calling these methods from CLI
    logging = logging.getLogger(__name__)
except:
    pass

class Indeed(object):
    def __init__(self, query_type):
        self.query_type = query_type
        self.add_loc = None
        self.stop_words = None
        self.num_samp = 1000
        self.df = pd.DataFrame(columns=['url', 'job_key', 'summary', 'summary_stem', 'city', 'jobtitle'])
        self.config_path = os.path.join(repo_dir, "tokens.cfg")
        self.query = None
        self.title = None
        self.stem_inverse = {}
        self.locations = None

    def _decode(self, string):
        try:
            string = string.encode('ascii', 'ignore').encode("ascii", "ignore")
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
        start = '&start=%(start)s'
        frm = '&fromage=360'
        limit = '&limit=25'
        site = '&st=jobsite'
        format = '&format=json'
        filter = '&filter=1'
        country = '&co=us'
        suffix = '&userip=1.2.3.4&useragent=Mozilla/%%2F4.0%%28Firefox%%29&v=2'

        self.api = prefix + pub + chan + loc + query + start + frm + limit + \
                   site + format + country + filter + suffix

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

    def _get_count(self):
        df = self.df.reset_index()
        df = self.summary_similarity(df, 'summary', 80)
        count = df.count()['url']
        return count

    def get_data(self, ind, start):
        data, num_res, end = self.get_url(start)
        logging.debug("number of results:%i" % num_res)
        num_res = np.int32(num_res)
        end = int(end)

        for item in data:
            content = self.get_content(item[0])
            parsed_content, soup = self.parse_content(content)
            if parsed_content is None:
                logging.debug("parsed content is None")
                continue
            self.df.loc[ind, 'url'] = item[0]
            self.df.loc[ind, 'city'] = item[1]
            self.df.loc[ind, 'jobtitle'] = item[2]
            self.df.loc[ind, 'job_key'] = item[3]
            self.df.loc[ind, 'summary'] = parsed_content
            self.df.loc[ind, 'summary_stem'] = self.stemmer_(parsed_content)
            self.df.loc[ind, 'grammar'] = self.get_grammar_content(content, soup)
            ind += 1
            logging.debug("index: %i" % ind)

        count = self._get_count()
        logging.debug("count: %i" % count)

        return ind, end, num_res, count

    def get_url(self, start):
        api = self.api %{'pub_id':self.pub_id,
                         'loc':'nationwide',
                         'channel_name':self.channel_name,
                         'query':self.format_keywords(self.query),
                         'start':start
                        }

        logging.debug("full api:%s" % api)

        try:
            response = urllib2.urlopen(api)
            data = json.loads(response.read())
            response.close()

            results = [(item['url'], item['city'], item['jobtitle'], item['jobkey'])  for item in data['results']]

        except urllib2.HTTPError, err:
            logging.debug("get url: %s" % err)
            return None

        except Exception, err:
            logging.debug("get url: %s" % err)
            return None

        return results, data['totalResults'], data['end']

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

    @classmethod
    def len_tester(self, word_list, thres=3):
        new_list = []
        for word in word_list:
            if len(word) < thres:
                continue
            else:
                new_list.append(word)

        return new_list

    def stemmer_(self, string):

        if string is None:
            return None

        string = self._decode(string)
        words = toker(string)
        words = map(lambda x:x.lower(), words)
        words = self.len_tester(words)
        stem_words = map(stemmer.stem, words)

        #master dict of stemmed words and originals
        for s,w in zip(stem_words, words):
            self.stem_inverse[s] = w

        return " ".join(stem_words)

    def _get_parsed_li(self, soup):
        obj = re.compile(r'summary|description')
        result = self._get_li(soup, 'span', obj)

        if result:
            return result
        else:
            result = self._get_li(soup, 'div', obj)
            if result:
                return result
            else:
                return None

    def _get_li(self, soup, div_, class_):
        class_data = soup.find(div_, {'class':class_})
        if class_data is not None:
            skills = class_data.find_all("li")
            output = [item.get_text() for item in skills]
            output = self.len_tester(output)

            if len(output) > 0:
                parsed = " ".join(output).replace('\n', '')
                return parsed
            else:
                corpus = class_data.get_text().replace('\n', '')
                return grammar.main(corpus)
        else:
            return False

    def parse_content(self, content):
        try:
            content = self._decode(content)
            soup = BeautifulSoup(content, 'html.parser')

            for obj in soup(['script', 'style', 'meta', 'a',
                             'input', 'img', 'noscript']):
                obj.extract()

            parsed = self._get_parsed_li(soup)

            return parsed, soup

        except Exception, err:
            logging.debug("soup didn't parse anything")
            logging.error(err)
            print err
            return None, None

    def get_grammar_content(self, content, soup):
        for obj in soup(['li']):
            obj.extract()

        re_obj = re.compile(r'summary|description')
        data = soup.find_all(['span', 'div', 'p'], {'class':re_obj})
        if data is None:
            return None

        text = [item.get_text() for item in data]
        text = " ".join(text)

        return grammar.main(text)

    def save_data(self):

        self.df.drop_duplicates(subset=['url'], inplace=True)
        self.df.dropna(inplace=True, how='any', axis=0)

        self.df.to_csv('/home/daniel/git/Python2.7/DataScience/indeed/data_frame.csv',
                        index=False,
                        encoding='utf-8')

    def main(self):
        self.load_config()
        self.build_api_string()
        self.add_stop_words()

    def vectorizer(self, corpus, max_features=200, max_df=0.8, min_df=5, n_min=2, n_max=3):
        vectorizer = CountVectorizer(max_features=max_features,
                                    max_df=max_df,
                                    min_df=min_df,
                                    lowercase=True,
                                    stop_words=self.stop_words,
                                    ngram_range=(n_min, n_max),
                                    analyzer='word',
                                    decode_error='ignore',
                                    strip_accents='unicode',
                                    binary=True
                                    )

        matrix = vectorizer.fit_transform(corpus)
        features = vectorizer.get_feature_names()

        return matrix, features

    def tfidf_vectorizer(self, corpus, max_features=100, max_df=0.8, min_df=0.1, n_min=2, n_max=3):
        vectorizer = TfidfVectorizer(max_features=max_features,
                                    max_df=max_df,
                                    min_df=min_df,
                                    lowercase=True,
                                    stop_words=self.stop_words,
                                    ngram_range=(n_min, n_max),
                                    analyzer='word',
                                    decode_error='ignore',
                                    strip_accents='unicode',
                                    sublinear_tf=True,
                                    binary=True,
                                    use_idf=False,
                                    norm=None
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

    def build_corpus_from_sent(self, keyword, column):
        keyword = stemmer.stem(keyword)

        documents = self.df[column].apply(lambda x: x.decode("utf-8", "ignore"))
        corpus = []
        for doc in documents:
            corpus.extend(tokenize.sent_tokenize(doc))

        new = []
        obj = re.compile(keyword, re.I)

        for sentence in corpus:
            if obj.search(sentence):
                new.append(sentence)

        return new

    def summary_similarity(self, df, column, ratio_thres):
        dup_list = []

        for i in range(df.shape[0] - 1):
            try:
                string1 = df.loc[i, column]
                string1 = self._decode(string1)
            except UnicodeDecodeError:
                continue

            for j in range(i+1, df.shape[0] - 1):
                try:
                    string2 = df.loc[j+1, column]
                    string2 = self._decode(string2)
                except UnicodeDecodeError:
                    continue

                ratio = fuzz.ratio(string1, string2)

                if ratio >= ratio_thres:
                    dup_list.append(j)

        return df.drop(df.index[dup_list])

    def clean_dup_words(self):
        self.obj = re.compile(r'(\b.+\b) \1\b')
        self.df['summary'] = self.df['summary'].apply(lambda x: self._clean_helper(x) if self.obj.search(x) else x)
        self.df['stem_summary'] = self.df['summary_stem'].apply(lambda x: self._clean_helper(x) if self.obj.search(x) else x)
        self.df['grammar' ] = self.df['grammar'].apply(lambda x: self._clean_helper(x) if self.obj.search(x) else x)

    def _clean_helper(self, x):
        reg = self.obj.search(x)
        words = reg.groups()
        for word in words:
            x = re.sub(word, '', x)

        return x


if __name__ == "__main__":
    ind = Indeed()
    ind.main()
