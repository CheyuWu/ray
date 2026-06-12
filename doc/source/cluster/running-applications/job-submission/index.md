(jobs-overview)=

# Ray Jobs Overview

Once you have deployed a Ray cluster (on [VMs](vm-cluster-quick-start) or [Kubernetes](kuberay-quickstart)), you are ready to run a Ray application!
![A diagram that shows the two primary ways to run a job on a Ray cluster.](../../images/ray-job-diagram.png "Two ways to run a job on a Ray cluster.")

## Ray Jobs API

The recommended way to run a job on a Ray cluster is to use the *Ray Jobs API*, which consists of a CLI tool, Python SDK, and a REST API.

The Ray Jobs API allows you to submit locally developed applications to a remote Ray Cluster for execution.
It simplifies the experience of packaging, deploying, and managing a Ray application.

A submission to the Ray Jobs API consists of:

1. An entrypoint command, like `python my_script.py`, and
2. A [runtime environment](runtime-environments), which specifies the application's file and package dependencies.

A job can be submitted by a remote client that lives outside of the Ray Cluster.
We will show this workflow in the following user guides.

After a job is submitted, it runs once to completion or failure, regardless of the original submitter's connectivity.
Retries or different runs with different parameters should be handled by the submitter.
Jobs are bound to the lifetime of a Ray cluster, so if the cluster goes down, all running jobs on that cluster will be terminated.

### Automatic in-place retries

Optionally, you can configure a job to be retried automatically on failure by passing a `retry_policy`.
When set, a retryable driver failure re-runs the driver **in place** using the same `submission_id` and the **same Ray cluster** (after an exponential backoff), instead of failing immediately.
This lets the cluster's own fault tolerance recover worker failures while the job simply re-submits the driver, without recreating the cluster.

For example, with the Python SDK:

```python
client.submit_job(
    entrypoint="python train.py",
    retry_policy={
        "max_retries": 3,
        "backoff": {
            "initial_delay_seconds": 30,
            "max_delay_seconds": 600,
            "multiplier": 2.0,
        },
        "retry_on": ["NON_ZERO_EXIT"],
        "no_retry_on_exit_codes": [130],
    },
)
```

The current attempt is observable via the `attempt_number` field of the job status (0 for the original run, 1 for the first retry, and so on).

```{note}
`DRIVER_OOM` is detected on a best-effort basis from exit code 137 (`128 + SIGKILL`), which is indistinguishable from an external `SIGKILL`.
Head node failures are not covered by `retry_policy`; recovering from those still requires restarting the Ray cluster.
```

To get started with the Ray Jobs API, check out the [quickstart](jobs-quickstart) guide, which walks you through the CLI tools for submitting and interacting with a Ray Job.
This is suitable for any client that can communicate over HTTP to the Ray Cluster.
If needed, the Ray Jobs API also provides APIs for [programmatic job submission](ray-job-sdk) and [job submission using REST](ray-job-rest-api).

## Running Jobs Interactively

If you would like to run an application *interactively* and see the output in real time (for example, during development or debugging), you can:

- (Recommended) Run your script directly on a cluster node (e.g. after SSHing into the node using [`ray attach`](ray-attach-doc)), or
- (For Experts only) Use [Ray Client](ray-client-ref) to run a script from your local machine while maintaining a connection to the cluster.

Note that jobs started in these ways are not managed by the Ray Jobs API, so the Ray Jobs API will not be able to see them or interact with them (with the exception of `ray job list` and `JobSubmissionClient.list_jobs()`).

## Contents

```{toctree}
:maxdepth: '1'

quickstart
sdk
jobs-package-ref
cli
rest
ray-client
```
