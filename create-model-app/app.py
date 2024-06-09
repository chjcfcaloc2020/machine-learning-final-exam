from shiny import App, ui, reactive, render
from shiny.types import FileInfo
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import threading
import numpy as np
import itertools

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_file("csv", "Choose csv file:", accept=[".csv"], multiple=False),
        ui.input_selectize("x", "X variables", choices=[], multiple=True),
        ui.input_selectize("y", "Y variable", choices=[], multiple=False),
        ui.hr(),
        ui.input_radio_buttons(
            "models", 
            "Choose model:",
            {
                "linear": "Linear Regression",
                "logistic": "Logistic Regression",
                "knn": "KNN Regression",
                "tree": "Decision Trees",
            }
        ),
    ),  
    ui.navset_tab(
        ui.nav_panel("Plot", ui.output_plot("plot")),
        ui.nav_panel("Accuracy & Equation", ui.output_text("accuracy")),
        ui.nav_panel("Confusion Matrix", ui.output_plot("confusion_matrix"))
    )
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
    
    @render.plot
    def plot():
        df = parsed_file()
        if df.empty:
            return
        
        x_columns = input.x()
        y_column = input.y()
        model_name = input.models()

        for x_column in x_columns:
            if model_name == "linear":
                model = LinearRegression()
            elif model_name == "logistic":
                model = LogisticRegression()
            elif model_name == "knn":
                model = KNeighborsRegressor()
            elif model_name == "tree":
                model = DecisionTreeRegressor()
            
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

    @render.text
    def accuracy():
        df = parsed_file()
        if df.empty:
            return "No data loaded"
        
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

    @render.plot
    def confusion_matrix():
        df = parsed_file()
        if df.empty:
            return "No data loaded"
        
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
app = App(app_ui, server)
