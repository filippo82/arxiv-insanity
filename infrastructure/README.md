# Infrastructure

This directory contains the code for either building or provisioning the infrastructure required by the
various components of this project.

## CI/CD

The images defined by the various Dockerfiles present in the various subfolders
are built using Cloud Build. The build process is triggered by

This is the rough of steps which need to be performed in order to set up
the Cloud Build integration for GitHub:

* install Google Cloud Build app
* create a connection to GitHub (Repositories tab)
* link a specific repository on Cloud Build (Repositories tab)
* create trigger

## `prefect-docker-images`

The `prefect-docker-images` directory contains the files (such as Dockerfiles) required to build the images
used by `prefect` to run the flows either as Cloud Run Jobs or Kubernetes jobs.
