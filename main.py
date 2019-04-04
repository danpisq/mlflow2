

import os
import mlflow
import click

import mlflow
from mlflow.entities import RunStatus, Run
from mlflow.utils.logging_utils import eprint
import six

from mlflow.tracking.fluent import _get_experiment_id


def _get_params(run):
    """Converts [mlflow.entities.Param] to a dictionary of {k: v}."""
    return {param.key: param.value for param in run.data.params}


def _already_ran(entry_point_name, parameters, source_version, experiment_id=None):
    """Best-effort detection of if a run with the given entrypoint name,
    parameters, and experiment id already ran. The run must have completed
    successfully and have at least the parameters provided.
    """
    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    client = mlflow.tracking.MlflowClient()
    all_run_infos = reversed(client.list_run_infos(experiment_id))
    for run_info in all_run_infos:
        if run_info.entry_point_name != entry_point_name:
            continue

        full_run = client.get_run(run_info.run_uuid)
        run_params = _get_params(full_run)
        match_failed = False
        for param_key, param_value in six.iteritems(parameters):
            run_value = run_params.get(param_key)
            if run_value != param_value:
                match_failed = True
                break
        if match_failed:
            continue

        if run_info.status != RunStatus.FINISHED:
            eprint(("Run matched, but is not FINISHED, so skipping "
                    "(run_id=%s, status=%s)") % (run_info.run_uuid, run_info.status))
            continue
        if run_info.source_version != source_version:
            eprint(("Run matched, but has a different source version, so skipping "
                    "(found=%s, expected=%s)") % (run_info.source_version, source_version))
            continue
        return client.get_run(run_info.run_uuid)
    return None


# TODO(aaron): This is not great because it doesn't account for:
# - changes in code
# - changes in dependant steps
def _get_or_run(entrypoint, parameters, source_version, use_cache=True):
    existing_run = _already_ran(entrypoint, parameters, source_version)
    if use_cache and existing_run:
        print("Found existing run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
        return existing_run
    print("Launching new run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters)
    return mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)


@click.command()
@click.option("--data-file", default="wine-quality.csv")
@click.option("--alpha")
@click.option("--l1-ratio")
def main(data_file, alpha, l1_ratio):
	with mlflow.start_run() as active_run:
		source_version = active_run.info.source_version
		load_etl_run = _get_or_run("etl", {}, source_version)
		
		etl_data_uri = os.path.join(load_etl_run.info.artifact_uri, "training_data.csv")

		train_run = _get_or_run("train", {"alpha":alpha, "l1_ratio":l1_ratio, "data-file":etl_data_uri}, source_version)


if __name__ == '__main__':
	main()