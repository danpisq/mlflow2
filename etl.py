import os
import click
import mlflow
import tempfile
import pandas as pd
from sklearn.preprocessing import StandardScaler



def normalize_features(data_frame, target_col):
	'''
		Normalizes features in dataframe excluding target_col
	'''
	features_df = data_frame.drop(columns=[target_col], axis=1)

	scaler = StandardScaler()
	scaled_df = pd.DataFrame(scaler.fit_transform(features_df), columns=features_df.columns)
	scaled_df[target_col] = data_frame[target_col]
	return scaled_df


def transform(data_frame):
	#normalize
	scaled_df = normalize_features(data_frame, "quality")
	#other transformations
	return scaled_df


@click.command()
@click.option("--training_data", default="./wine-quality.csv" )
def etl_data(training_data):
	with mlflow.start_run():
		transformed_df = transform( pd.read_csv(training_data) )

		mlflow.log_param("norm", "MinMaxScaler")
		mlflow.log_metric("cols", transformed_df.shape[1])
		mlflow.log_metric("rows", transformed_df.shape[0])

		tmpdir = tempfile.mkdtemp()
		training_data_path = os.path.join(tmpdir, 'training_data.csv')
		transformed_df.to_csv(training_data_path)
		mlflow.log_artifact(training_data_path)
		
		training_desc_path = os.path.join(tmpdir, 'training_desc.txt')
		transformed_df.describe().to_csv(training_desc_path)
		mlflow.log_artifact(training_desc_path)

		
if __name__ == '__main__':
	etl_data()	