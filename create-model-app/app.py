from shiny import App, ui, reactive, render
from shinyswatch import theme
from shiny.types import FileInfo
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from scipy.sparse.linalg import svds
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split
from pathlib import Path
from scipy.sparse.linalg import svds
import threading
import numpy as np

# Call css file
css_file = Path(__file__).parent / "css" / "styles.css"

# UI
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_file("csv", "Choose csv file:", accept=[".csv"], multiple=False),
        ui.input_selectize("x", "X variables", choices=[], multiple=True, options=(
            {"placeholder": "Enter the X variables",}
        )),
        ui.input_selectize("y", "Y variable", choices=[], multiple=False , options=(
            {"placeholder": "Enter the Y variable",}
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
        ui.hr(),
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
        ui.nav_panel("Confusion Matrix", ui.output_plot("plot_confusion_matrix")),
        ui.nav_panel("Recommendation Models", ui.output_table("recommendation")),
        ui.nav_panel("Recommendation Books", 
            ui.row(
                ui.column(4, ui.input_file("books", "Choose books csv file:", accept=[".csv"], multiple=False)),
                ui.column(4, ui.input_file("users", "Choose users csv file:", accept=[".csv"], multiple=False)),
                ui.column(4, ui.input_file("ratings", "Choose ratings csv file:", accept=[".csv"], multiple=False)),
            ),
            ui.output_table("recommendations"),
        )
    ),
    ui.include_css(css_file),
    theme.darkly(),
)

# Xử lý data
def process_data(df):
    imputer = SimpleImputer(strategy='mean')
    df[df.select_dtypes(include=['float64', 'int64']).columns] = imputer.fit_transform(df.select_dtypes(include=['float64', 'int64']))
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_cols] = StandardScaler().fit_transform(df[numeric_cols])
    return df

# Hiển thị phương trình
def display_equation(model):
    if hasattr(model, "coef_"):  
        equation = "Coefficients: {}\nIntercept: {}".format(model.coef_, model.intercept_)
    elif hasattr(model, "feature_importances_"): 
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

    # load books scv
    @reactive.calc
    def load_books_csv():
        file: list[FileInfo] | None = input.books()
        if file is None:
            return pd.DataFrame()

        books = pd.read_csv(file[0]["datapath"], sep=";", on_bad_lines='skip', low_memory=False, encoding="latin-1")
        books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']

        books.drop(['imageUrlS', 'imageUrlM', 'imageUrlL'], axis=1, inplace=True)
        books = books[(books.yearOfPublication != 'DK Publishing Inc') & (books.yearOfPublication != 'Gallimard')]
        books.yearOfPublication = books.yearOfPublication.astype('int32')
        books = books.dropna(subset=['publisher'])

        return books

    @reactive.calc
    def load_users_csv():
        file: list[FileInfo] | None = input.users()
        if file is None:
            return pd.DataFrame()

        users = pd.read_csv(file[0]["datapath"], sep=";", on_bad_lines='skip', low_memory=False, encoding="latin-1")
        users.columns = ['userID', 'Location', 'Age']

        users.loc[(users.Age > 90) | (users.Age < 5), 'Age'] = np.nan
        users.Age = users.Age.fillna(users.Age.mean())
        users.Age = users.Age.astype(np.int32)

        return users

    @reactive.calc
    def load_ratings_csv():
        file: list[FileInfo] | None = input.ratings()
        if file is None:
            return pd.DataFrame()

        ratings = pd.read_csv(file[0]["datapath"], sep=";", on_bad_lines='skip', low_memory=False, encoding="latin-1")
        ratings.columns = ['userID', 'ISBN', 'bookRating']

        return ratings
    
    # preview navbar
    @render.text
    def error_text():
        if input.csv() is None:
            return "No data loaded. Please choose a CSV file!"
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

        # Validation
        if not x_columns or not y_column:
            return
        if not all(col in df.columns for col in x_columns) or y_column not in df.columns:
            return

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
                model = RandomForestRegressor(random_state=0, n_estimators=40)
                
            if model:
                # Convert y_column to discrete labels if model = logistic
                if model_name == "logistic":
                    df[y_column], _ = pd.factorize(df[y_column])
                model_thread = threading.Thread(target=model.fit, args=(df[[x_column]], df[y_column]))
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
            return "No data loaded. Please choose a CSV file!"
        
        x_columns = input.x()
        y_column = input.y()
        model_name = input.models()
        
        # Debug prints
        print(f"x_columns: {x_columns}")
        print(f"y_column: {y_column}")
        print(f"model_name: {model_name}")

        # Validation
        if not x_columns or not y_column:
            return "Please select both X and Y variables."
        if not all(col in df.columns for col in x_columns) or y_column not in df.columns:
            return "Selected columns are not in the DataFrame."

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
            model = RandomForestRegressor(random_state=0, n_estimators=40)

        if model:
            X_train, X_test, y_train, y_test = train_test_split(df[list(x_columns)], df[y_column], test_size=0.3, random_state=0)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if model_name == "logistic":
                accuracy = accuracy_score(y_test, y_pred)
                equation = display_equation(model)
                return f"Accuracy: {accuracy}\n{equation}"
            else:
                y_test_discrete = np.round(y_test)
                y_pred_discrete = np.round(y_pred)
                accuracy = accuracy_score(y_test_discrete, y_pred_discrete)
                mse = mean_squared_error(y_test, y_pred)
                equation = display_equation(model)
                return f"Accuracy: {accuracy}\nMean Squared Error: {mse}\n{equation}"

    # plot_confusion_matrix navbar
    @render.plot
    def plot_confusion_matrix():
        df = parsed_file()
        if df.empty:
            return
        
        x_columns = input.x()
        y_column = input.y()
        model_name = input.models()
        
        # Validation
        if not x_columns or not y_column:
            return
        if not all(col in df.columns for col in x_columns) or y_column not in df.columns:
            return
        
        model = None
        
        if model_name == "linear":
            return "Confusion matrix is not applicable for linear regression"
        elif model_name == "logistic":
            model = LogisticRegression()
        elif model_name == "knn":
            model = KNeighborsRegressor()
        elif model_name == "tree":
            model = DecisionTreeRegressor()
        elif model_name == "forest":
            model = RandomForestRegressor(random_state=0, n_estimators=40)
        
        if model:
            X_train, X_test, y_train, y_test = train_test_split(df[list(x_columns)], df[y_column], test_size=0.3, random_state=0)
            
            if isinstance(y_test.iloc[0], (int, float)):
                # Nếu biến mục tiêu là liên tục, chuyển thành rời rạc
                y_train_discrete = np.round(y_train).astype(int)
                y_test_discrete = np.round(y_test).astype(int)
            else:
                y_train_discrete = y_train
                y_test_discrete = y_test
            
            model.fit(X_train, y_train_discrete)
            y_pred = model.predict(X_test)
            
            if isinstance(y_pred[0], (float)):
                y_pred = np.round(y_pred).astype(int)
            
            cm = confusion_matrix(y_test_discrete, y_pred)
            
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            tick_marks = np.arange(len(set(y_test_discrete)))
            plt.xticks(tick_marks, tick_marks)
            plt.yticks(tick_marks, tick_marks)
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()

    @render.table
    def recommendation():
        df = parsed_file()
        if df.empty:
            return pd.DataFrame({"Model": ["No model selected"], "Accuracy": [], "Mean Squared Error": [], "Coefficients/Feature Importances": [], "Intercept": []})

        x_columns = input.x()
        y_column = input.y()
        model_name = input.models()

        # Validation
        if not x_columns or not y_column:
            return pd.DataFrame({"Model": ["No model selected"], "Accuracy": [], "Mean Squared Error": [], "Coefficients/Feature Importances": [], "Intercept": []})
        if not all(col in df.columns for col in x_columns) or y_column not in df.columns:
            return pd.DataFrame({"Model": ["No model selected"], "Accuracy": [], "Mean Squared Error": [], "Coefficients/Feature Importances": [], "Intercept": []})

        results = []

        for model_name in ["linear", "logistic", "knn", "tree", "forest"]:
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
                model = RandomForestRegressor(random_state=0, n_estimators=40)

            if model:
                try:
                    X_train, X_test, y_train, y_test = train_test_split(df[list(x_columns)], df[y_column], test_size=0.3, random_state=0)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    if model_name == "logistic":
                        accuracy = accuracy_score(y_test, y_pred)
                        equation = display_equation(model)
                        results.append({
                            "Model": f"Logistic Regression",
                            "Accuracy": accuracy,
                            "Mean Squared Error": "N/A",
                            "Coefficients/Feature Importances": equation,
                            "Intercept": "N/A"
                        })
                    else:
                        y_test_discrete = np.round(y_test)
                        y_pred_discrete = np.round(y_pred)
                        accuracy = accuracy_score(y_test_discrete, y_pred_discrete)
                        mse = mean_squared_error(y_test, y_pred)
                        equation = display_equation(model)
                        results.append({
                            "Model": f"{model_name.capitalize()} Regression",
                            "Accuracy": accuracy,
                            "Mean Squared Error": mse,
                            "Coefficients/Feature Importances": equation,
                            "Intercept": "N/A"
                        })

                except ValueError as e:
                    if "Unknown label type: continuous" in str(e):
                        continue

        if not results:
            return pd.DataFrame({"Model": ["No model selected"], "Accuracy": [], "Mean Squared Error": [], "Coefficients/Feature Importances": [], "Intercept": []})
        else:
            df_results = pd.DataFrame(results)
            df_results = df_results.sort_values(by=["Accuracy", "Mean Squared Error"], ascending=[False, True])
            return df_results
    
    # recommender books
    @render.table
    def recommendations():
        books_df = load_books_csv()
        users_df = load_users_csv()
        ratings_df = load_ratings_csv()

        if books_df.empty or users_df.empty or ratings_df.empty:
            return pd.DataFrame()
        
        n_users = users_df.shape[0]
        n_books = books_df.shape[0]

        ratings_new = ratings_df[ratings_df.ISBN.isin(books_df.ISBN)]
        ratings_explicit = ratings_new[ratings_new.bookRating != 0]
        ratings_implicit = ratings_new[ratings_new.bookRating == 0]

        counts1 = ratings_explicit['userID'].value_counts()
        ratings_explicit = ratings_explicit[ratings_explicit['userID'].isin(counts1[counts1 >= 100].index)]

        ratings_matrix = ratings_explicit.pivot(index='userID', columns='ISBN', values='bookRating').fillna(0)
        userID = ratings_matrix.index
        ISBN = ratings_matrix.columns

        U, sigma, Vt = svds(ratings_matrix.to_numpy(), k=50)
        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
        preds_df = pd.DataFrame(all_user_predicted_ratings, columns=ratings_matrix.columns)

        user_id = 2
        userID = ratings_matrix.iloc[user_id-1, :].name

        sorted_user_predictions = preds_df.iloc[user_id].sort_values(ascending=False)

        user_data = ratings_explicit[ratings_explicit.userID == (userID)]
        book_data = books_df[books_df.ISBN.isin(user_data.ISBN)]
        user_full_info = user_data.merge(book_data)

        recommendations = (books_df[~books_df['ISBN'].isin(user_full_info['ISBN'])].
                   merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left', left_on = 'ISBN'
                         ,right_on = 'ISBN')).rename(columns = {user_id: 'Predictions'})

        return recommendations.sort_values('Predictions', ascending = False).iloc[:10, :]

app = App(app_ui, server)
