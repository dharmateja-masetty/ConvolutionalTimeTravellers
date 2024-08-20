import numpy as np  
import pandas as pd 
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
#import os
#from dotenv import load_dotenv
#from sqlalchemy import create_engine



"""def get_data(query_string, df_name):
      
        '''Execute a SQL query on a PostgreSQL database, save the result as a CSV file,
        and return it as a pandas DataFrame.

        Parameters:
        query_string (str): SQL query string to execute on the database.
        df_name (str): Name to use for saving the CSV file and DataFrame.

        Returns:
        pd.DataFrame: DataFrame containing the result of the SQL query.

        Raises:
        Exception: If there's an issue connecting to the database or executing the query.

        Notes:
        - Requires environment variables DATABASE, USER_DB, PASSWORD, HOST, and PORT to be set via dotenv.
        - Saves the CSV file in a 'data' directory relative to the current working directory.
'''
        try:
            load_dotenv()  # Load environment variables from .env file
            DATABASE = os.getenv('DATABASE')
            USER_DB = os.getenv('USER_DB')
            PASSWORD = os.getenv('PASSWORD')
            HOST = os.getenv('HOST')
            PORT = os.getenv('PORT')
            
            # Construct the database connection string
            DB_STRING = f"postgresql://{USER_DB}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
            
            # Create engine to connect to the database
            db = create_engine(DB_STRING)
            
            # Create 'data' directory if it doesn't exist
            if not os.path.exists("./data"):
                os.mkdir("data")
            
            # Read data from database using SQLAlchemy
            df_sqlalchemy = pd.read_sql(query_string, db)
            
            # Save DataFrame as CSV file
            df_sqlalchemy.to_csv(f"data/{df_name}.csv", index=False)
            
            # Read CSV file into pandas DataFrame
            df = pd.read_csv(f"data/{df_name}.csv")
            
            return df
        
        except Exception as e:
            raise Exception(f"Error fetching data from database: {str(e)}")
"""



def analyze_data(df, y):
        """
        Perform comprehensive data inspection including displaying data, descriptive statistics, 
        checking for duplicates, missing values, plotting target variable, and detecting outliers.

        Parameters:
        df (pd.DataFrame): The input DataFrame to inspect.
        y (pd.Series): The target variable Series.

        Returns:
        None: This function displays various insights and visualizations about the dataset.

        """
        print("Display some values from the dataframe:")
        display(df.head())
        print("-"*10)
        print(f"Dataframe Shape: {df.shape}")
        print("-"*10)
        print("Dataframe Descriptive Statistics:")
        display(df.describe().T)
        print("-"*10)
        print("Unique values from each column:")
        print(df.nunique())
        print("-"*10)
        check_duplicates(df)
        print("-"*10)
        # Plot the missing values
        plot_missing_values(df)
        print("-"*10)
        # Print percentages of missing values 
        print(percentage_of_missing_values(df))
        print("-"*10)
        # Plot the target variable
        plot_target_variable(y)
        print("-"*10)
        print(f"Checking for data imbalance:\n{y.value_counts()}")
        print("-"*10)
        # Check for outliers
        #detect_outliers(df, y)


def plot_missing_values(df):
        """
        Plot missing values in the DataFrame and show percentage of missing values per column.

        Parameters:
        df (pd.DataFrame): The input DataFrame to plot missing values.

        Returns:
        None: This function plots the missing value distribution and prints the percentage of missing values per column.
        """
        fig, axs = plt.subplots(1, 2, figsize=(18, 6))  # Increase figsize for more space
        
        # Plotting missing value bar chart
        axs[0].set_title("Distribution of Missing Values")
        msno.bar(df, ax=axs[0])
        
        # Plotting missing value matrix
        axs[1].set_title("Matrix of Missing Values")
        msno.matrix(df,sparkline=False, ax=axs[1])
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to prevent overlapping
        plt.show()

def percentage_of_missing_values(df):
        # Calculate the percentage of missing values per column
        missing_values_count = df.isna().sum()
        total_values = df.shape[0]  # Total number of rows in the DataFrame
        percentage_missing_values = (missing_values_count / total_values) * 100
        # Display the percentages of missing values per column
        print("Percentage of missing values per column:")
        return(percentage_missing_values)


def plot_target_variable(y):
        """
        Plot the distribution of the target variable.

        Parameters:
        y (pd.Series): The target variable Series to plot.

        Returns:
        None: This function plots the distribution of the target variable.
        """
        print("Plotting the target variable")
        # Plot the target variable separately
        plt.figure(figsize=(8, 6))
        sns.countplot(x=y)
        plt.title("Distribution of Target Variable")
        plt.xlabel("Target Variable")
        plt.ylabel("Count")
        plt.grid(False)
        plt.show()
        

def check_duplicates(data):
        """
        Check for duplicates in a DataFrame.

        Parameters:
        data (pd.DataFrame): The DataFrame to check for duplicates.

        Returns:
        None: Prints the number of duplicates found if any, otherwise prints "No duplicates found !!!".
        """
        print("Checking for duplicates")
        has_dup = data.duplicated()
        true_dup = np.where(has_dup == True)
        if len(true_dup[0]) > 0:
            print("Data has", len(true_dup[0]), "duplicates")
        else:
            print("No duplicates found !!!")

def detect_outliers(df, y):
        """
        Detect outliers in the DataFrame features.

        Parameters:
        df (pd.DataFrame): The input DataFrame to detect outliers.
        y (pd.Series): The target variable Series.

        Returns:
        None: This function plots boxplots for each feature in the DataFrame to detect outliers.
        """
        print("Checking for outliers")
        feature_list = df.columns.tolist()
        feature_list.remove(y.name)
        df_outliers = df[feature_list].copy()
        df_outliers.plot(kind='box', subplots=True, layout=(8,3), figsize=(34,30))
        plt.show()

def fill_missing_values(df, missing_dict):
        """
        Fill missing values in DataFrame columns based on specified methods.

        Parameters:
        df (pd.DataFrame): The DataFrame containing missing values.
        missing_dict (dict): Dictionary where keys are column names with missing values
                            and values are the method to use for filling missing values.
                            Valid methods include 'mean', 'median', 'mode', 'ffill', 'bfill',
                            or a specific value (e.g., 0).

        Returns:
        pd.DataFrame: DataFrame with missing values filled based on the specified methods.
        """
        for column, method in missing_dict.items():
            if method == 'mean':
                df[column].fillna(df[column].mean(), inplace=True)
            elif method == 'median':
                df[column].fillna(df[column].median(), inplace=True)
            elif method == 'mode':
                df[column].fillna(df[column].mode()[0], inplace=True)
            elif method == 'ffill':
                df[column].fillna(method='ffill', inplace=True)
            elif method == 'bfill':
                df[column].fillna(method='bfill', inplace=True)
            else:
                df[column].fillna(method, inplace=True)  # Specific value provided

        return df

def cleaning_data(df, rename_dict=None,columns_to_drop=None,drop_duplicates=True,missing_dict=None):
        """
        Clean and preprocess a DataFrame by renaming columns, dropping specified columns,
        dropping duplicates, and filling missing values based on provided parameters.

        Parameters:
        df (pd.DataFrame): The DataFrame to be cleaned.
        rename_dict (dict, optional): Dictionary mapping old column names to new names. Default is None.
        columns_to_drop (list of str, optional): List of column names to drop. Default is None.
        drop_duplicates (bool, optional): Whether to drop duplicate rows. Default is True.
        missing_dict (dict, optional): Dictionary where keys are column names with missing values
                                    and values are the method to use for filling missing values.
                                    Valid methods include 'mean', 'median', 'mode', 'ffill', 'bfill',
                                    or a specific value. Default is None.

        Returns:
        None: Modifies the input DataFrame `df` in place.
        """
        if rename_dict is not None:
            df.rename(columns=rename_dict, inplace=True)
        if columns_to_drop is not None:
            df = df.drop(columns=columns_to_drop)
        if drop_duplicates:
            df = df.drop_duplicates()
        if missing_dict is not None:
            df= fill_missing_values(df,missing_dict)