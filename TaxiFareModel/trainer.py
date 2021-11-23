# imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from sklearn.model_selection import train_test_split
from TaxiFareModel.data import get_data, clean_data

class Trainer():
    def __init__(self, X=None, y=None):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
        ('dist_trans', DistanceTransformer()),
        ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
            ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
            ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('linear_model', LinearRegression())
            ])
        self.pipeline = pipe
        return self

    def train(self, X_train, y_train):
        self.pipeline.fit(X_train,y_train)
        return self

    def run(self):
        """set and train the pipeline"""
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                              ('stdscaler', StandardScaler())])
        time_pipe = Pipeline([('time_enc',
                               TimeFeaturesEncoder('pickup_datetime')),
                              ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
            "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
            'dropoff_longitude'
        ]), ('time', time_pipe, ['pickup_datetime'])],
                                         remainder="drop")
        pipe = Pipeline([('preproc', preproc_pipe),
                         ('linear_model', LinearRegression())])
        self.pipeline = pipe.fit(self.X,self.y)
        return self

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)

        rmse = compute_rmse(y_pred, y_test)
        return rmse


if __name__ == "__main__":
    df = get_data()
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    trainer = Trainer()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

    trainer.set_pipeline()
    trainer.train(X_train,y_train)

    rmse = trainer.evaluate(X_val, y_val)

    print(rmse)
    print('TODO')
