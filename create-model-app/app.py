from shiny import App, ui, reactive, render
from shinyswatch import theme
from shiny.types import FileInfo
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from pathlib import Path
import threading
import numpy as np
import itertools
import io

#call css file
css_file = Path(__file__).parent / "css" / "styles.css"

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_file("csv", "Choose csv file:", accept=[".csv"], multiple=False),
        ui.input_selectize("x", "X variables", choices=[], multiple=True, options=(
            { "placeholder": "Enter the X variables", }
        )),
        ui.input_selectize("y", "Y variable", choices=[], multiple=False , options=(
            { "placeholder": "Enter the Y variable", }
        )),
        ui.hr(),
        ui.input_radio_buttons(
            "models", 
            "Choose model:",
            {
                "linear": "Linear Regression",
                "logistic": "Logistic Regression",
                "knn": "KNN Regression",
                "tree": "Decision Trees",
                "forest": "Random Forest",
            }
        ),
    ),  
    ui.navset_card_pill(
        ui.nav_panel("Preview", 
            ui.output_text("error_text"),
            ui.output_text("data_preview"),
            ui.output_table("data_preview_tbl"),
            ui.output_text("describe"),
            ui.output_table("describe_tbl"),
            ui.output_text("data_missing"),
            ui.output_table("missing_tbl"),
        ),
        ui.nav_panel("Plot", ui.output_plot("plot")),
        ui.nav_panel("Accuracy & Equation", ui.output_text("accuracy")),
        ui.nav_panel("Confusion Matrix", ui.output_plot("confusion_matrix"))
    ),
    ui.include_css(css_file),
    theme.darkly(),
)

def process_data(df):
    imputer = SimpleImputer(strategy='mean')
    df[df.select_dtypes(include=['float64', 'int64']).columns] = imputer.fit_transform(df.select_dtypes(include=['float64', 'int64']))
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])
    return df

def train_model(model, X, y, output):
    model.fit(X, y)

def calculate_metrics(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    return mse

def display_equation(model):
    if hasattr(model, "coef_"):  # Kiểm tra xem mô hình có thuộc tính coef_ không (dành cho hồi quy tuyến tính)
        equation = "Coefficients: {}\nIntercept: {}".format(model.coef_, model.intercept_)
    elif hasattr(model, "feature_importances_"):  # Kiểm tra xem mô hình có thuộc tính feature_importances_ không (dành cho cây quyết định)
        equation = "Feature Importances: {}".format(model.feature_importances_)
    else:
        equation = "Equation not available"
    return equation

# SERVER
def server(input, output, session):
    @reactive.calc
    def parsed_file():
        file: list[FileInfo] | None = input.csv()
        if file is None:
            return pd.DataFrame()
        
        df = pd.read_csv(file[0]["datapath"])
        df = process_data(df)
        
        numeric_cols = df.columns.tolist()
        ui.update_selectize("x", choices=numeric_cols)
        ui.update_selectize("y", choices=numeric_cols)
        
        return df
    
    # preview navbar
    @render.text
    def error_text():
        if input.csv() is None:
            return "No data loaded. Please chose csv file!"
        else:
            return ""
        
    @render.text
    def data_preview():
        if input.csv() is None:
            return ""
        else:
            return "Data Preview"

    @render.table
    def data_preview_tbl():
        df = parsed_file()
        if df is None:
            return pd.DataFrame()
        else:
            return df.head()

    @render.text
    def describe():
        df = parsed_file()
        if input.csv() is None:
            return ""
        else:
            return "Data Describe"
      
    @render.table
    def describe_tbl():
        df = parsed_file()
        if input.csv() is None:
            return pd.DataFrame()
        else:
            return df.describe()

    @render.text
    def data_missing():
        df = parsed_file()
        if input.csv() is None:
            return ""
        else:
            return "Missing Values"

    @render.table
    def missing_tbl():
        df = parsed_file()
        if input.csv() is None:
            return pd.DataFrame()
        else:
            missing_values = df.isnull().sum().reset_index()
            missing_values.columns = ['Column', 'Missing Values']
            return missing_values

    # plot navbar
    @render.plot
    def plot():
        df = parsed_file()
        if df.empty:
            return
        
        x_columns = input.x()
        y_column = input.y()
        model_name = input.models()

        for x_column in x_columns:
            model = None
            if model_name == "linear":
                model = LinearRegression()
            elif model_name == "logistic":
                model = LogisticRegression()
            elif model_name == "knn":
                model = KNeighborsRegressor()
            elif model_name == "tree":
                model = DecisionTreeRegressor()
            elif model_name == "forest":
                model = RandomForestRegressor(criterion="squared_error", random_state=0, n_estimators=40)
            
            if model:
                model_thread = threading.Thread(target=train_model, args=(model, df[[x_column]], df[y_column], output))
                model_thread.start()
                model_thread.join()
                
                plt.scatter(df[x_column], df[y_column], color='blue', label='Data')
                if model_name == "logistic":
                    plt.plot(df[x_column], model.predict_proba(df[[x_column]])[:,1], color='green', label='Logistic Regression')
                else:
                    plt.plot(df[x_column], model.predict(df[[x_column]]), color='green', label='{} Regression'.format(model_name.upper()))

        plt.xlabel(', '.join(x_columns))
        plt.ylabel(y_column)
        plt.title("{} Scatter Plot".format(model_name.capitalize()))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    # accuracy navbar
    @render.text
    def accuracy():
        df = parsed_file()
        if df.empty:
            return "No data loaded. Please chose csv file!"
        
        x_columns = input.x()
        y_column = input.y()
        model_name = input.models()
        model = None

        if model_name == "linear":
            model = LinearRegression()
        elif model_name == "logistic":
            model = LogisticRegression()
        elif model_name == "knn":
            model = KNeighborsRegressor()
        elif model_name == "tree":
            model = DecisionTreeRegressor()
        
        if model:
            return None
        
    # confusion_matrix navbar
    @render.plot
    def confusion_matrix():
        df = parsed_file()
        if df.empty:
            return "No data loaded. Please chose csv file!"
        
        x_columns = input.x()
        y_column = input.y()
        model_name = input.models()
        model = None
        
        if model_name == "linear":
            model = LinearRegression()
        elif model_name == "logistic":
            model = LogisticRegression()
        elif model_name == "knn":
            model = KNeighborsRegressor()
        elif model_name == "tree":
            model = DecisionTreeRegressor()
        elif model_name == "forest":
            model = RandomForestClassifier(n_estimators=10, criterion="entropy")
        
        if model:
            return None    
app = App(app_ui, server)
