import sys
import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, fbeta_score, make_scorer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
nltk.download(['punkt', 'wordnet', 'stopwords'])

"""
The pipeline of Machine learning, which is to train, test, improve model. Finally, it will store a 
model which could be used by Web-app
"""


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    A class that be used to create a pipeline of model
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def load_data(database_filepath):
    """
    Args:
    Loading a data from database that we have created at ETL pipeline

    Return:
    X: The dependent variable that we want to predict
    Y: The independent variables that we use to predict the X
     category_names

    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disater_response', engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'], axis=1)
    column_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    """
    Args:
    List of text in english

    Return:
    clean_tokens: which has been process

    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Return:
    A model that we have to use to predict data

    """

    # pipeline = Pipeline([
    #     ('vect', CountVectorizer(tokenizer=tokenize)),
    #     ('tfidf', TfidfTransformer()),
    #     ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    # try other machine learning algorithms, add other features besides the TF-IDF
    model = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer(use_idf=True))
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(
            random_state=None, n_estimators=150)))
    ])

    return model 


def evaluate_model(model, X_test, Y_test, category_names):
    """
    The function will evaluate the model via score of precision, recall and F1

    Args:
    model: A model that has trained to evaluate;
    X_test: A dataset that used to predict
    Y_test: A label that to be predict
    category_names: category's name

    """   
    # print the current score of model
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))
    
    # check parameter then decide which parameters have to be optimized 
    # model.get_params()
    
    # parameters = { 'clf__estimator__min_samples_split': [2,4],
    #            'clf__estimator__n_estimators': [10,100,150],
    #            'clf__estimator__random_state': [None, 42],
    #            'tfidf__use_idf':[True, False],
    #             }
    
    # cv = GridSearchCV(model, param_grid = parameters, verbose=10, cv=5)
    # # Warning: The following processing takes 10 hours in the cloud server.
    # cv.fit(X_train, y_train)
    
    # # get the score after optimizing
    # y_pred_optimized = cv.predict(X_test)
    # print(classification_report(Y_test, y_pred_optimized, target_names=category_names))
    pass

def save_model(model, model_filepath):
    """
    The function will create a model of pkl file that would be used in web-app

    Args:
    model: A model that has trained and evaluated;
    model_filepath: destination path to save .pkl file
    """   
    # save the model to disk
    filename = model_filepath+'.pkl'
    pickle.dump(model, open(filename, 'wb'))
    pass

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
