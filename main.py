from utils import *

import mlflow
import click


def workflow():
	with mlflow.start_run() as active_run:
		pass
       
if __name__ == '__main__':
	workflow()