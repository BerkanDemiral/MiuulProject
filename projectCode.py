import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

""""
#################### Veri Setine Temel Bakış ve EDA ###################################

**Ana Özellikler:**

- **Age:** Age of the gym member.
- **Gender:** Gender of the gym member (Male or Female).
- **Weight (kg):** Member’s weight in kilograms.
- **Height (m):** Member’s height in meters.
- **Max_BPM:** Maximum heart rate (beats per minute) during workout sessions.
- **Avg_BPM:** Average heart rate during workout sessions.
- **Resting_BPM:** Heart rate at rest before workout.
- **Session_Duration (hours):** Duration of each workout session in hours.
- **Calories_Burned:** Total calories burned during each session.
- **Workout_Type:** Type of workout performed (e.g., Cardio, Strength, Yoga, HIIT).
- **Fat_Percentage:** Body fat percentage of the member.
- **Water_Intake (liters):** Daily water intake during workouts.
- **Workout_Frequency (days/week):** Number of workout sessions per week.
- **Experience_Level:** Level of experience, from beginner (1) to expert (3).
- **BMI:** Body Mass Index, calculated from height and weight.

"""

df = pd.read_csv("dataset/gym_members_exercise_tracking.csv")
df_ = df.copy() 
df.head()

df.Workout_Type.unique()
df.info()

### BOŞ DEĞERLER ###
df.isnull().any()

#############################################
# Kategorik Değişken Analizi 
#############################################
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
num_but_cat = [col for col in df.columns if df[col].nunique() < 5 and df[col].dtypes in ["int", "float"]]
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]
df[cat_cols].nunique()
cat_cols

df[cat_cols]

def cat_summary(dataframe, col_name, plot=False):
    """
    Veri setindeki kategorik değişkenleri bize verir. Bunu yaparken aslında kategorik gibi görünmeyen ama kategorik olması gerekenleri de tespit ederiz. 

    Parameters
    ----------
    dataframe: dataframe
        değişkenleri alacağımız df
    col_name: string
        df içerisindeki kategorik olduğunu bildiğimiz değişkenlerdir
    plot: bool
        Bir görselleştirme yapmak istiyor musun? Eğer istiyorsak True olarak yollarız. Kategorikleri için baktığımız için basit olarak countplot dönecektir

    Notes
    ------
    İlgili kategorik değişkendeki unique ifadelerin sayıları, ve o değişken içerisindeki yüzdesini verir

    """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#########################################################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    if df[col].dtypes == "bool":
        print("Bool Değişken Kategorik Değildir")
    else:
        cat_summary(df, col, plot=True)


#############################################
# Sayısal Değişken Analizi 
#############################################


num_cols = [col for col in df.columns if df[col].dtypes in ["int","float"]]
num_cols = [col for col in num_cols if col not in cat_cols]


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """
    Bir veri setindeki bir sütunun aykırı değer sınırlarını belirler.

    Parameters
    ----------
    dataframe : DataFrame
        Aykırı değer sınırlarının belirleneceği veri seti.
    col_name : str
        Aykırı değer analizinin yapılacağı sütun adı.
    q1 : float, optional
        Çeyreklik değeri; genellikle 0.25 olarak alınır (default: 0.25).
    q3 : float, optional
        Çeyreklik değeri; genellikle 0.75 olarak alınır (default: 0.75).

    Returns
    -------
    low_limit : float
        Aşağı aykırı değer sınırı.
    up_limit : float
        Yukarı aykırı değer sınırı.

    Notes
    ------
    Fonksiyon, verilen sütundaki aykırı değerleri bulmak için IQR (Interquartile Range) yöntemini kullanır.
    1. İlk olarak, sütunun belirtilen q1 ve q3 çeyreklik değerlerini hesaplar.
    2. IQR (çeyreklikler arası mesafe) hesaplanır.
    3. Aykırı değerlerin sınırlarını belirlemek için 1.5 * IQR kuralı kullanılır:
       - Aşağı sınır (low_limit) = q1 - 1.5 * IQR
       - Yukarı sınır (up_limit) = q3 + 1.5 * IQR
    Bu sınırların dışında kalan değerler aykırı değer olarak kabul edilir.
    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


import matplotlib.pyplot as plt
import seaborn as sns
import warnings

def num_summary(dataframe, numerical_col, plot=False):
    """
    Veri setindeki sayısal değişkenleri bize verir

    Parameters
    ----------
    dataframe: dataframe
        değişkenleri alacağımız df
    col_name: string
        df içerisindeki kategorik olduğunu bildiğimiz değişkenlerdir
    plot: bool
        Bir görselleştirme yapmak istiyor musun? Eğer istiyorsak True olarak yollarız. Kategorikleri için baktığımız için basit olarak countplot dönecektir

    Notes
    ------
    İlgili sayısal değişkendeki unique ifadelerin sayıları, ve o değişken içerisindeki yüzdesini verir
    Ek olarak aykırı değer analizini de gerçekleştirir. 
    Değişken bazlı olarak 0.25 ve 0.75 banında IQR tespiti yaparak aykırı değerler tespit edilir. Grafikte gösterilir ve bu değerlerin neler olduğu yazılır. 

    """
    warnings.filterwarnings("ignore", category=FutureWarning)

    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    summary = dataframe[numerical_col].describe(quantiles).T
    print(f"{numerical_col} özet istatistikleri:")
    print(summary)

    lower_bound, upper_bound = outlier_thresholds(dataframe, numerical_col)

    outliers = dataframe[(dataframe[numerical_col] < lower_bound) | (dataframe[numerical_col] > upper_bound)]
    num_outliers = outliers.shape[0]
    outlier_values = outliers[numerical_col].values
    print(f"\n{numerical_col} sütununda {num_outliers} adet aykırı değer bulunmaktadır.")
    print(f"Aykırı değerler: {outlier_values}")

    if plot:
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        sns.histplot(dataframe[numerical_col], kde=True, color='skyblue')
        plt.axvline(lower_bound, color='red', linestyle='--', label='Alt Sınır')
        plt.axvline(upper_bound, color='red', linestyle='--', label='Üst Sınır')
        plt.xlabel(numerical_col)
        plt.title(f"{numerical_col} Histogram ve Yoğunluk Grafiği (Aykırı Değer Sınırları)")
        plt.legend()

        plt.subplot(1, 2, 2)
        sns.boxplot(x=dataframe[numerical_col], color='lightcoral')
        plt.axvline(lower_bound, color='red', linestyle='--', label='Alt Sınır')
        plt.axvline(upper_bound, color='red', linestyle='--', label='Üst Sınır')
        plt.xlabel(numerical_col)
        plt.title(f"{numerical_col} Kutu Grafiği (Aykırı Değer Sınırları)")
        plt.legend()

        plt.tight_layout()
        plt.show()

    return outliers


for col in num_cols:
    num_summary(df, col, plot=True)
    
####################### Aykırı Değerlere Müdahale ###########################
    

def fill_outliers_with_median(dataframe, col_name):
    """
    Aykırı değerleri medyan değeri ile doldurur.

    Parameters
    ----------
    dataframe : DataFrame
        Aykırı değerlerin doldurulacağı veri seti.
    col_name : str
        Aykırı değerleri doldurulacak sütun adı.

    Returns
    -------
    DataFrame
        Aykırı değerlerin medyan ile doldurulduğu güncellenmiş veri seti.
    """
    # Eşik değerleri hesapla
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    
    # Medyan değeri hesapla
    median_value = dataframe[col_name].median()
    
    # Aykırı değerleri medyan ile doldur
    dataframe.loc[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit), col_name] = median_value
    return dataframe


fill_outliers_with_median(df,"BMI")

fill_outliers_with_median(df,"Calories_Burned")

fill_outliers_with_median(df,"Weight (kg)")


num_summary(df,"BMI",True)
num_summary(fill_outliers_with_median(df,"Calories_Burned"),"Calories_Burned",True)
num_summary(fill_outliers_with_median(df,"Weight (kg)"),"Weight (kg)",True)


#############################################
# Korelasyon Analizi (Analysis of Correlation)
#############################################

corr = df[num_cols].corr()

sns.set(rc={'figure.figsize': (10, 10)})
sns.heatmap(corr, cmap="RdBu")
plt.show()



################################### Encoding ############################

from sklearn.preprocessing import LabelEncoder
encoder_gender = LabelEncoder()
encoder_workout = LabelEncoder()

df["Gender"] = encoder_gender.fit_transform(df["Gender"])
df["Workout_Type"] = encoder_workout.fit_transform(df["Workout_Type"])

df.head()

# Cardio -> 0
# HIIT -> 1
# Strength -> 2
# Yoga ->3

# Male -> 0
# Female -> 1


df.drop("Workout_Frequency (days/week)", axis = 1 , inplace= True)

df.head()

df.drop("Experience_Level", axis = 1 , inplace= True)

df.head()

####################################### ## Machine Learning ###################################


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

X = df.drop(columns=['Calories_Burned'])
y = df['Calories_Burned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

# 2. KNN
knn = KNeighborsRegressor()
knn_params = {'n_neighbors': [3, 5, 7, 9, 11]}
grid_knn = GridSearchCV(knn, knn_params, cv=5, scoring='neg_mean_squared_error')
grid_knn.fit(X_train_scaled, y_train)
best_knn = grid_knn.best_estimator_
y_pred_knn = best_knn.predict(X_test_scaled)

# 3. Gradient Boosting
gb = GradientBoostingRegressor(random_state=42)
gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}
grid_gb = GridSearchCV(gb, gb_params, cv=5, scoring='neg_mean_squared_error')
grid_gb.fit(X_train, y_train)
best_gb = grid_gb.best_estimator_
y_pred_gb = best_gb.predict(X_test)

models = {
    "Linear Regression": y_pred_lr,
    "KNN": y_pred_knn,
    "Gradient Boosting": y_pred_gb
}

for name, y_pred in models.items():
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: RMSE = {rmse:.2f}, R2 = {r2:.2f}")



######################### Standart Scaler ######################################### 
    

df.head()

scaler = StandardScaler()
numerical_features = df.select_dtypes(include=["int64","float64"]).columns
df[numerical_features] = scaler.fit_transform(df[numerical_features])

X = df.drop(columns=['Calories_Burned'])  
y = df['Calories_Burned'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(max_depth=5, min_samples_split=10),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
    'KNeighbors Regressor': KNeighborsRegressor(n_neighbors=5, weights='uniform'),
    'Support Vector Regressor': SVR(kernel='rbf', C=1.0, epsilon=0.1),
    'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
}

best_model_name = None
best_model = None
best_rmse_score = float('inf')  

model_score = []
for name, model in models.items():
    model.fit(X_train, y_train)  

    y_pred = model.predict(X_test)  

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    model_score.append((name, mse, rmse, r2, mae))

    print(f"{name} - MSE: {mse}")
    print(f"{name} - RMSE: {rmse}")
    print(f"{name} - R²: {r2}")
    print(f"{name} - MAE: {mae}\n")

    if rmse < best_rmse_score:  
        best_rmse_score = rmse
        best_model_name = name
        best_model = model

print(f"Best Model: {best_model_name} with RMSE score of {best_rmse_score}")


model_names = [score[0] for score in model_score]
mse_values = [score[1] for score in model_score]
r2_values = [score[2] for score in model_score]
mae_values = [score[3] for score in model_score]

x = range(len(model_names))

plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
plt.bar(x, mse_values, color='blue')
plt.xticks(x, model_names, rotation=45)
plt.title('Mean Squared Error (MSE)')
plt.ylabel('MSE')

plt.subplot(1, 4, 2)
plt.bar(x, mse_values, color='blue')
plt.xticks(x, model_names, rotation=45)
plt.title('Root Mean Squared Error (RMSE)')
plt.ylabel('RMSE')

plt.subplot(1, 4, 3)
plt.bar(x, r2_values, color='green')
plt.xticks(x, model_names, rotation=45)
plt.title('R² Score')
plt.ylabel('R²')

plt.subplot(1, 4, 4)
plt.bar(x, mae_values, color='orange')
plt.xticks(x, model_names, rotation=45)
plt.title('Mean Absolute Error (MAE)')
plt.ylabel('MAE')

plt.tight_layout()
plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grids = {
    'Decision Tree Regressor': {
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5, 10]
    },
    'Random Forest Regressor': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'random_state': [42]
    },
    'KNeighbors Regressor': {
        'n_neighbors': [3, 5, 10],
        'weights': ['uniform', 'distance']
    },
    'Support Vector Regressor': {
        'C': [0.1, 1, 10],
        'epsilon': [0.1, 0.2, 0.5]
    },
    'Gradient Boosting Regressor': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
}

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(),
    'Random Forest Regressor': RandomForestRegressor(),
    'KNeighbors Regressor': KNeighborsRegressor(),
    'Support Vector Regressor': SVR(),
    'Gradient Boosting Regressor': GradientBoostingRegressor()
}

best_models = {}
for name, model in models.items():
    print(f"Tuning {name}...")
    if name in param_grids:
        grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Best CV MSE for {name}: {np.sqrt(-grid_search.best_score_)}")
    else:
        model.fit(X_train, y_train)
        best_models[name] = model

for name, model in best_models.items():
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    print(f"Test RMSE for {name}: {rmse}")
