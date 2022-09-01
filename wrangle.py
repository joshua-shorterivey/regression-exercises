
#### imports

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from env import user, password, host
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

import warnings
warnings.filterwarnings("ignore")


#### Acquire
def acquire_zillow():
    #create url to access DB
    url = f"mysql+pymysql://{user}:{password}@{host}/zillow"

    #write sql query for specified columns
    sql = """ 
    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
    FROM properties_2017
    LEFT JOIN propertylandusetype USING (propertylandusetypeid)
    WHERE propertylandusedesc IN ("Single Family Residential", "Inferred Single Family Residential")
    """

    #read the data in a dataFrame
    df = pd.read_sql(sql, url)

    # take care of column names --> no need for takeaways
    df = df.rename(columns = {'bedroomcnt':'bedrooms',
                                'bathroomcnt': 'bathrooms',
                                'calculatedfinishedsquarefeet': 'area',
                                'taxvaluedollarcnt': 'tax_value',
                                'taxamount': 'tax_amount',
                                'yearbuilt': 'year_built'})
    return df

#### Prepare (remove outliers, do some visualizations, and train/test/impute)

def remove_outliers(df, k, col_list):
    """ 
    Purpose
        Remove outliers from a list of columns in a dataframe and return that dataframe
    
    Parameters
        df: a dataframe containing zillow real estate data
        k: factor to multiple IQR
        col_list: a list of dataframe columns to work on 

    Returns
        df: a dataframe with the desired adjustments
    """

    # total number of observations
    num_obs = df.shape[0]
        
    # Create a column that will label our rows as containing an outlier. sets default value
    df['outlier'] = False

    # loop through the columns provided to find appropriate values and labels
    for col in col_list:

        # find quartiles
        q1, q3 = df[col].quantile([.25, .75])  
        
       # get interquartile range
        iqr = q3 - q1

       # find upper/lower bounds 
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # update the outlier label as needed. 
        df['outlier'] = np.where(((df[col] < lower_bound) | (df[col] > upper_bound)) & (df.outlier == False), True, df.outlier)
    
    # set dataframe to dataframe w/o the outliers
    df = df[df.outlier == False]

    # drop the outlier column from the dataFrame. no longer needed
    df.drop(columns=['outlier'], inplace=True)

    # print out number of removed observations
    print(f"Number of observations removed: {num_obs - df.shape[0]}")
        
    return df

def make_hist(df):
    plt.figure(figsize=(20,4))
    i=1
    for col in df.drop(columns=['fips', 'year_built']).columns:

        plt.subplot(1, len(df.columns), i)
        df[col].hist(bins=5)
        plt.title(col)
        plt.ticklabel_format(useOffset=False)
        plt.tight_layout()
        i += 1

    plt.show()

def make_box(df):
    # boxplot 
    plt.figure(figsize=(20,4))
    i=1

    for col in df.drop(columns=['fips', 'year_built']).columns:
    
        plt.subplot(1, len(df.columns), i)
        sns.boxplot(data=df[[col]])
        plt.title(col)
        i += 1

    plt.show()

# prep function to bring it together
def prep_zillow(df):
    """
    Purpose
        To return dataset for exploration

    Parameters
        df: dataframe to perform desired operations on

    Returns
        train, validate, and test datasets
    """
    #remove outliers
    df = remove_outliers(df, 1.5, ['bedrooms', 'bathrooms', 'area', 'tax_value', 'tax_amount'])

    # view distributions with histogram and boxplots
    make_hist(df)
    make_box(df)

    # change data types for fips and year_built. 
    df.fips = df.fips.astype(object)
    df.year_built = df.year_built.astype(object)

    #train_test_split
    train_validate, test = train_test_split(df, test_size=.2, random_state=514)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=514)

    # loop throught continous columsn and impute with median
    for col in df.drop(columns=['year_built', 'fips']).columns:
        imputer = SimpleImputer(strategy='median')

        imputer.fit(train[[col]])

        train[[col]] = imputer.transform(train[[col]])
        validate[[col]] = imputer.transform(validate[[col]])
        test[[col]] = imputer.transform(test[[col]])

    #create imputer for cat types 
    imputer_cat = SimpleImputer(strategy='most_frequent')

    #fit both imputers to train set
    imputer_cat.fit(train[['year_built']])

    train[['year_built']] = imputer_cat.transform(train[['year_built']])
    validate[['year_built']] = imputer_cat.transform(validate[['year_built']])
    test[['year_built']] = imputer_cat.transform(test[['year_built']])

    return train, validate, test

#### Wrangle

def wrangle_zillow():
    """ 
    Purpose
        Acquire and prepare data from Zillow data set for exploration
    
    Parameters
        None
    
    Returns 
        Cleaned and prepared train, validate, and test subsets of data for exploration and modeling
    """
    train, validate, test = prep_zillow(acquire_zillow())
    return train, validate, test