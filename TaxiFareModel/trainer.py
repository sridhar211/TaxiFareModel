# imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from TaxiFareModel.utils import compute_rmse, haversine_vectorized
from TaxiFareModel.data import get_data, clean_data
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property

MLFLOW_URI = "https://mlflow.lewagon.ai/"


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.experiment_name = "[DE][Remote][sridhar211]TaxiFareModel_1.0"
        self.X = X
        self.y = y

    def set_pipeline(self):
        '''returns a pipelined model'''
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
        self.pipeline=pipe
        return

    def run(self):
        """set and train the pipeline"""
        '''returns a trained pipelined model'''
        self.pipeline.fit(self.X, self.y)
        return

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

if __name__ == "__main__":
    # get and clean data
    samples=1000
    df = get_data(samples)
    df = clean_data(df)

    # set X and y
    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

    trainer=Trainer(X_train,y_train)

    # build pipeline
    trainer.set_pipeline()

    # train the pipeline
    trainer.run()

    # evaluate the pipeline
    rmse = trainer.evaluate(X_val, y_val)

    print(rmse)

    trainer.mlflow_log_metric("rmse", rmse)
    trainer.mlflow_log_param("model", 'linear')
    trainer.mlflow_log_param("Train_samples", samples)
