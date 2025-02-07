###################################################
# Setup                                           #
###################################################

#--------------------------------------------------
# import packages

import os
from os.path import join
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

#--------------------------------------------------
# paths

main_path = Path(os.getcwd()).parents[0]
data_path = join(main_path, "data")

###################################################
# Reading                                         #
###################################################

#--------------------------------------------------
# load data

training_data = pd.read_csv(
    join(
        data_path,
        "raw",
        "titanic_train.csv"
    )    
)

test_data = pd.read_csv(
    join(
        data_path,
        "raw",
        "titanic_test.csv"
    )    
)

# NOTE: this data reflects a prediction that all women survived and all men
# died. Since there is no other information available, I use this as this is the
# actual survival pattern
survived_test = pd.read_csv(
    join(
        data_path,
        "raw",
        "titanic_gender_submission.csv"
    )    
)

#--------------------------------------------------
# ratio between training and test data

test_data.shape[0] / (test_data.shape[0] + training_data.shape[0])

# Answer: 31.9%

###################################################
# Preparation                                     #
###################################################

#--------------------------------------------------
# renaming the columns for more clarity

def renaming_columns(titanic_data: pd.DataFrame) -> pd.DataFrame:
    """
    Renames some of the columns in the dataset.

    Some of the columns do not have clear names and therefore, are adjusted.    

    Parameters
    ----------
    titanic_data : pd.DataFrame
        Dataset with Titanic information (can be training or test dataset).

    Returns
    -------
    titanic_data_copy : pd.DataFrame
        Modified dataset with renamed columns.
        
    Raises
    ------
    KeyError
        If the required columns (to be renamed) are not in the dataset.

    """
    
    #--------------------------------------------------
    # check if required columns are in the dataset
    required_columns = ["Pclass", "SibSp", "Parch"]
    for col in required_columns:
        if col not in titanic_data.columns:
            raise KeyError(f"Missing required column: '{col}'")

    #--------------------------------------------------
    titanic_data_copy = titanic_data.copy()
    
    # transform column names into lowercase
    titanic_data_copy.columns = [col.lower() for col in titanic_data_copy.columns]
    
    # rename some columns
    titanic_data_copy = titanic_data_copy.rename(columns = {
        "pclass": "passenger_class",
        "sibsp": "num_siblings_spouses",
        "parch": "num_parents_children"
    })
    
    #--------------------------------------------------
    # return
    
    return titanic_data_copy

#--------------------------------------------------
# feature engineering and cleaning

def adding_feature_family_size(titanic_data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the family size feature to the dataset.
    
    The family size is calculated as the sum of the number of siblings/ spouses
    and the number of parents/ children aboard the Titanic, including the
    passenger themselves.

    Parameters
    ----------
    titanic_data : pd.DataFrame
        Dataset with Titanic information (can be training or test dataset).
        Must contain 'num_siblings_spouses' and 'num_parents_children' columns.

    Returns
    -------
    titanic_data : pd.DataFrame
        Modified dataset with additional 'family_size' column.
        
    Raises
    ------
    KeyError
        If the required columns are missing from the dataset.

    """
    
    #--------------------------------------------------
    # ensure that necessary columns are present
    
    required_columns = ["num_siblings_spouses", "num_parents_children"]
    for col in required_columns:
        if col not in titanic_data.columns:
            raise KeyError(f"Missing required column: '{col}'")
    
    #--------------------------------------------------
    # define family size as combination of number of siblings/ spouses and number of
    # parents and children
    
    titanic_data["family_size"] = (
        titanic_data["num_siblings_spouses"] +
        titanic_data["num_parents_children"] +
        1
    )
    
    #--------------------------------------------------
    # return
    
    return titanic_data

def handling_missings(titanic_data: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing in the 'fare' and 'age' columns
    
    Handles missing in the 'fare' and 'age' columns by replacing them with the
    median value.

    Parameters
    ----------
    titanic_data : pd.DataFrame
        Dataset with Titanic information (can be training or test dataset).

    Returns
    -------
    titanic_data_copy : pd.DataFrame
        Dataset where missings in 'fare' have been replaced with the median.

    """
    print(f"Number of missings in 'fare': {titanic_data['fare'].isna().sum()}")
    print(f"Number of missings in 'age': {titanic_data['age'].isna().sum()}")
    
    titanic_data_copy = titanic_data.copy()
    
    # replace missings with median
    titanic_data_copy.fillna(
        {
            "fare": titanic_data_copy["fare"].median(),
            "age": titanic_data_copy["age"].median()
        },
        inplace = True
    )
    
    #--------------------------------------------------
    # return
    
    return titanic_data_copy

def cleaning_titles(titanic_data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning titles
    
    This function extracts the titles of the passengers.

    Parameters
    ----------
    titanic_data : pd.DataFrame
        Dataset with Titanic information (can be training or test dataset).

    Raises
    ------
    ValueError
        If the title feature has missings, i.e. no title has been extracted.

    Returns
    -------
    titanic_data_copy : TYPE
        Modified dataset with the title column.

    """

    #--------------------------------------------------       
    # create auxiliary name column as lowercase
    
    titanic_data_copy = titanic_data.copy()
    titanic_data_copy["name_lowercase"] = titanic_data_copy["name"].str.lower()

    #--------------------------------------------------
    # define conditions and choices for titles
    
    conditions = [
        (
            titanic_data_copy["name_lowercase"].str.contains(r"\b(lady)\b") |
            titanic_data_copy["name_lowercase"].str.contains(r"\b(dona)\b") |
            titanic_data_copy["name_lowercase"].str.contains(r"\b(countess)\b") |
            titanic_data_copy["name_lowercase"].str.contains(r"\b(capt)\b") |
            titanic_data_copy["name_lowercase"].str.contains(r"\b(col)\b") |
            titanic_data_copy["name_lowercase"].str.contains(r"\b(don)\b") |
            titanic_data_copy["name_lowercase"].str.contains(r"\b(dr)\b") |
            titanic_data_copy["name_lowercase"].str.contains(r"\b(major)\b") |
            titanic_data_copy["name_lowercase"].str.contains("rev.") |
            titanic_data_copy["name_lowercase"].str.contains(r"\b(sir)\b") |
            titanic_data_copy["name_lowercase"].str.contains(r"\b(jonkheer)\b")
        ),
        (
            (titanic_data_copy["sex"] == "female") & (
                titanic_data_copy["name_lowercase"].str.contains("mlle") |
                titanic_data_copy["name_lowercase"].str.contains("ms.") |
                titanic_data_copy["name_lowercase"].str.contains("miss.")
            )
        ),
        (
            (titanic_data_copy["sex"] == "female") &(
                titanic_data_copy["name_lowercase"].str.contains("mme") |
                titanic_data_copy["name_lowercase"].str.contains("mrs.")
            )
        ),
        (
            (titanic_data_copy["sex"] == "male") & (
                titanic_data_copy["name_lowercase"].str.contains("mr.")
            )
        ),
        (
            titanic_data_copy["name_lowercase"].str.contains(r"\b(master)\b")    
        )
    ]

    choices = [
        "rare_title",
        "miss",
        "mrs",
        "mr",
        "master"
    ]

    # set titles
    titanic_data_copy["title"] = np.select(conditions, choices, default = None)
    
    # drop auxiliary column again
    titanic_data_copy.drop(columns = ["name_lowercase"], inplace = True)
    
    #--------------------------------------------------
    # check that there is title for each person
    
    if titanic_data_copy["title"].isna().sum() != 0:
        raise ValueError("Not all persons have an assigned title!")
    
    #--------------------------------------------------
    # return
    
    return titanic_data_copy


def encoding_categorical_features(titanic_data: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """
    Encoding categorical features
    
    This function encodes categorical and binary variables

    Parameters
    ----------
    titanic_data : pd.DataFrame
        Dataset with Titanic information (can be training or test dataset).
    categorical_cols : list
        List with categorical variables that should be encoded.

    Returns
    -------
    encoded_df : pd.DataFrame
        Dataframe that only contains the encoded variables.

    """
    
    #--------------------------------------------------
    # specify encoder for multi-categorical
    categorical_encoder = OneHotEncoder(sparse_output = False)
    
    # fit encoder
    encoded_categorical = categorical_encoder.fit_transform(titanic_data[categorical_cols])
    
    # transform into dataframe
    encoded_df = pd.DataFrame(
        encoded_categorical,
        columns = categorical_encoder.get_feature_names_out(categorical_cols)
    )
    
    #--------------------------------------------------
    # specify encoder for binary variable
    binary_encoder = LabelEncoder()
    
    # fit encoder
    encoded_binary = binary_encoder.fit_transform(titanic_data["sex"])
    
    # assign binary encoded variable to dataframe
    encoded_df = encoded_df.assign(sex = encoded_binary)
    
    # return
    return encoded_df

#--------------------------------------------------
# chain all cleaning together

def cleaning_pipeline(titanic_data: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    # apply transformations
    out = (titanic_data
        .pipe(renaming_columns)
        .pipe(adding_feature_family_size)
        .pipe(handling_missings)
        .pipe(cleaning_titles)
    )

    # encoded columns as dataframe
    encoded = encoding_categorical_features(
        out,
        categorical_cols = categorical_cols
    )

    # drop variables that have been encoded and are not needed
    out.drop(
        columns = ["embarked", "sex"],
        inplace = True
    )

    # combine both datasets
    out = pd.concat([out, encoded], axis = 1)
    
    # return
    return out

training_data_cleaned = cleaning_pipeline(
    titanic_data = training_data,
    categorical_cols = ["passenger_class", "embarked", "title"]
)

test_data_cleaned = cleaning_pipeline(
    titanic_data = test_data,
    categorical_cols = ["passenger_class", "embarked", "title"]
)

###################################################
# Prepare survived test data                      #
###################################################

# rename columns according to the other data
survived_test_cleaned = survived_test.rename(
    columns = {
        "PassengerId": "passengerid",
        "Survived": "survived"
    }    
)

# merge survived data to test data
test_data_cleaned = pd.merge(
    test_data_cleaned,
    survived_test_cleaned,
    on = "passengerid",
    how = "left"    
)


###################################################
# Export                                          #
###################################################

def exporting_cleaned_data(titanic_data: pd.DataFrame, filename: str):
    titanic_data.to_csv(
        join(
            data_path,
            "processed",
            filename + ".csv"
        ),
        index = False
    )

exporting_cleaned_data(training_data_cleaned, filename = "titanic_training_prep")
exporting_cleaned_data(test_data_cleaned, filename = "titanic_test_prep")
