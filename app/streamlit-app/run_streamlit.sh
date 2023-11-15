#!/usr/bin/env bash

PORT=8502

GKE_CLUSTER_IP=$(kubectl get service/vespa -o jsonpath='{.status.loadBalancer.ingress[*].ip}')
GKE_CLUSTER_PORT=$(kubectl get service/vespa -o jsonpath='{.spec.ports[?(@.name=="container")].port}')
export VESPA_ENDPOINT="http://${GKE_CLUSTER_IP}:${GKE_CLUSTER_PORT}"
echo $VESPA_ENDPOINT

streamlit run app.py --server.port=${PORT} --browser.serverAddress=0.0.0.0 \
    --server.headless=true \
    --theme.base=light \
    --server.runOnSave=true
