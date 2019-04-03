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
	scaled_df[target_col] = data_frame['target_col']
	return scaled_df


def transform(data_frame):
	#normalize
	scaled_df = normalize_features(data_frame)
	#other transformations
	return scaled_df


@click.command()
@click.option("--training_data", default="./wine-quality.csv" )
def etl_data(training_data):
	transformed_df = transform( pd.load_csv(training_data) )
	with mlflow.start_run():
		tmpdir = tempfile.mldtemp()
		training_data_path = os.path.join(tmpdir, 'training_data.csv')
		mlflow.log_artifacts(training_data_path, "training_data")

if __name__ == '__main__':
	etl_data()	