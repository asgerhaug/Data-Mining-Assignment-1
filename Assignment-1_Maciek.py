from influxdb import InfluxDBClient # install via "pip install influxdb"
import pandas as pd


client = InfluxDBClient(host='influxus.itu.dk', port=8086, username='lsda', password='icanonlyread')
client.switch_database('orkney')

def get_df(results):
    values = results.raw["series"][0]["values"]
    columns = results.raw["series"][0]["columns"]
    df = pd.DataFrame(values, columns=columns).set_index("time")
    df.index = pd.to_datetime(df.index) # Convert to datetime-index
    return df

# Get the last 90 days of power generation data
generation = client.query(
    "SELECT * FROM Generation where time > now()-90d"
    ) # Query written in InfluxQL

# Get the last 90 days of weather forecasts with the shortest lead time
wind  = client.query(
    "SELECT * FROM MetForecasts where time > now()-90d and time <= now() and Lead_hours = '1'"
    ) # Query written in InfluxQL

gen_df = get_df(generation)
wind_df = get_df(wind)


#drop the columns that are irrelevant
#I realize that I could have done it in a pipeline using ColumnTransformeR, yet I wanted to do it right away. 
# I also could not forsee why it would be better to include it in a pipeline.
gen_df = gen_df.drop(columns=["ANM", "Non-ANM"])



#Joinning dataframes on inner join which results in handling of missing data in terms of just abandoning it. 
#This is due t o no other features available that could help to determine the wind speed and directions at given time. 
# Also, I do not have any domain knowledge, 
# but I suppose taking an average wind speed would not be a good idea in this scenario as they probably fluctate a lot . 

final_df = gen_df.join(wind_df, on='time', how="inner")



from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt

#TODO: drop irrelevant columns, encode direction as cathegorical, scale data

#define X and y

y = final_df[['Total']]
X = final_df.drop(["Total"], axis=1)
#X = final_df[['Direction', 'Speed']]

#split data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

cat_col = ['Direction', 'Lead_hours']
num_col = ['Source_time', 'Speed']

cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

num_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler())])

pre_processor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_col),
        ('cat', cat_transformer, cat_col)
        #,("Drop Irrelevant", ColumnTransformer([
            #("Drop ANM", "drop", ["ANM"]),
            #('Drop Non - ANM', "drop",["Non-ANM"]), 
            #],remainder="passthrough"))
            ])

final_pipeline = Pipeline(steps=[('pre_processor', pre_processor),
                                 ('lin_reg', LinearRegression())])

final_pipeline.fit(X_train, y_train)

y_hat = final_pipeline.predict(X_test)

# Evaluating the model

#reset index to organize data
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

#plot the data
plt.plot(X_test.index, y_test, "--", label = "Truth")
plt.plot(X_test.index, y_hat, label = "Predictions")
plt.xlabel("Time")
plt.ylabel("MegaWatts")
plt.legend()
plt.show()

#calculate MAE
score = mean_absolute_error(y_test, y_hat)
print('MAE:', score)


# Get all future forecasts regardless of lead time
forecasts  = client.query(
    "SELECT * FROM MetForecasts where time > now()"
    ) # Query written in InfluxQL
for_df = get_df(forecasts)

# Limit to only the newest source time
newest_source_time = for_df["Source_time"].max()
newest_forecasts = for_df.loc[for_df["Source_time"] == newest_source_time].copy()
# Preprocess the forecasts and do predictions in one fell swoop 
# using your pipeline
final_pipeline.predict(newest_forecasts)




