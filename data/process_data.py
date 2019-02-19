import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

"""
The file is to process data from CSV file to sqlite 
"""


def load_data(messages_filepath, categories_filepath):
    """
    Args: 
       Two CSV files that is raw data what we have to process:
       The first CSV file containing messages (disaster_messages.csv)
       The second CSV file containing categories (disaster_categories.csv)

    Return:
       The Dataframe which has merge the two dataset above

    """
    # load the message data
    messages = pd.read_csv(messages_filepath)

    # load the categories data
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages, categories, on='id')

    return df


def clean_data(df):
    """
    Args: 
       A dataframe which merges two dataset loaded from csv.file

    Return:
       A dataframe which has been processed 
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.loc[1]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    print(category_colnames)

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(np.int)

    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat((df, categories), axis=1)

    df = process_outlier(df, categories)

    # check number of duplicates of all row then clean
    df[df.duplicated()].shape
    df = df.drop_duplicates(keep='first')
    df[df.duplicated()].shape

    return df


def process_outlier(df, categories):
    """
    Args: 
       A dataframe which merges two dataset loaded from csv.file

    Return:
       A dataframe which cleaned the outlier data
    """
    for col in categories.columns:
        print(df[col].unique())

    # check the unique value
    df.related.unique()
    df.related.replace(2, 1, inplace=True)

    return df


def save_data(df, database_filename):
    """
    Args: 
       A dataframe which has been cleaned
       The name of database that we want to store

    Return:
       A dataframe which cleaned the outlier data
    """

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disater_response', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
