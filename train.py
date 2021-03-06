import pandas as pd
import numpy as np
import click
import mlflow
import mlflow.sklearn
import sys
import os

from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

def eval_metrics(actual, pred):
	rmse = np.sqrt(mean_squared_error(actual, pred))
	mae = mean_absolute_error(actual, pred)
	r2 = r2_score(actual, pred)
	return rmse, mae, r2

@click.command()
@click.option("--run-id")
@click.option("--alpha", default=0.1)
@click.option("--l1-ratio", default=0.1)
def train(run_id , alpha, l1_ratio):
	with mlflow.start_run():
		
		try:
			mlflow_run = mlflow.tracking.MlflowClient().get_run(run_id)
			print("Got MlFlow run with id {}".format(run_id))
		except:
			sys.exit("\nEXCEPTION, no run with id {}\n".format(run_id))

		train_data_path = os.path.join(mlflow_run.info.artifact_uri, "training_data.csv")
		#print(train_data_path)
		df = pd.read_csv(train_data_path)
		#print(df.head())
		train, test = train_test_split(df)

		X_train = train.drop(['quality'], axis=1)
		X_test = test.drop(['quality'], axis=1)
		y_train = train['quality']
		y_test = test['quality']


		model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
		model.fit(X_train, y_train)

		predicted_qualities = model.predict(X_test)
		
		(rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

		print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
		print("  RMSE: %s" % rmse)
		print("  MAE: %s" % mae)
		print("  R2: %s" % r2)

		mlflow.log_param("alpha", alpha)
		mlflow.log_param("l1_ratio", l1_ratio)
		mlflow.log_metric("rmse", rmse)
		mlflow.log_metric("r2", r2)
		mlflow.log_metric("mae", mae)

		mlflow.sklearn.log_model(model, "model")



if __name__ == '__main__':
	train()