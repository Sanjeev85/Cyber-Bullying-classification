from collections import Counter
import operator
import pandas
import seaborn as sns
import re
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from nltk.stem.snowball import PorterStemmer, SnowballStemmer
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
stop_words = set(stopwords.words('english'))
x = set(STOPWORDS)
new_stop = set(['rt', ':)', 'i\'v', '\'', 'u', 'i\'m', '&amp;', '#mkr', 'know', '.'])
stop_words.update(x, new_stop)
label = {
    'not_cyberbullying' : 0,
    'gender':1,
    'religion':2,
    'age':3,
    'ethnicity':4,
    'other_cyberbullying':5
}



class GeneratePlots:
    '''
        Class used for genering word cloud and freq count bar graph
    '''

    def generateWordCloud(self, lbl: str, data: pandas.DataFrame, h=250, w=400, fs=55):
        ''''
            @:param1 = label of data
            @:param2 = main dataFrame
            @:param3 = height of word cloud
            @:param4 = width of word cloud
            @:param5 = max font size of word cloud text
        '''
#         print(label[lbl])
        type_df = data[data.cyberbullying_type == label[lbl]]
        text_data = ' '.join(list(type_df.tweet_text))
        word_cloud = WordCloud(height=h, width=w, max_font_size=fs).generate(text_data)
        plt.figure(figsize=(20, 10))
        plt.imshow(word_cloud)
        plt.show()
        return text_data

    
    def freqPlotter(self, string: str, label: str):
        ''''
        @:param1 = tweet_text in string format
        @:param2 = label required to plot
        '''
        toList = string.split()
        counter = Counter(toList)
        counter = dict(counter)
        counter = sorted(counter.items(), key=operator.itemgetter(1), reverse=True)
        # dictionary sorted in ascending order
        # most used words for gener based cyberbulling
        words = []
        freq = []
        c = 0
        for key, value in counter:
            if c == 15:
                break
            words.append(key)
            freq.append(value)
            c += 1
        plot = sns.barplot(x=words, y=freq)
        plot.set_xticklabels(plot.get_xticklabels(), rotation=90, ha="right")
        plt.tight_layout()
        plt.title(f'Frequently Used Words based on {label} Cyberbullying.')
        plt.show()


class Preprocess:
        def stem(self, sentence: str):
            ps = PorterStemmer()
            ss = SnowballStemmer('english')
#             lem = WordNetLemmatizer()
            sentence = ' '.join([ps.stem(word) for word in str(sentence).split()])
            sentence = ' '.join([ss.stem(word) for word in str(sentence).split()])
#             sentence = ' '.join([lem.lemmatize(word) for word in str(sentence).split()])
            return sentence
        
        def removeHttps(self, sentence: str):
            sentence : ' '.join([re.sub(r'http\S+', '', word) for word in str(sentence).split()])
            return sentence
    
        def removeStopwords(self, sentence : str):
            sentence = sentence.lower()
            sentence = ' '.join([word for word in str(sentence).split() if word not in stop_words])
            return sentence

        def markLabels(self, text : str):
            text = label[text]
            return text
