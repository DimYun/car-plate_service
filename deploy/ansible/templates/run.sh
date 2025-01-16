#!/bin/bash

set -ue

IMAGE={{ docker_image }}:{{ docker_tag }}

docker run \
    -d \
    -p {{ service_port }}:{{ service_port }} \
    --name={{ container_name }} \
    --restart always \
    ${IMAGE}