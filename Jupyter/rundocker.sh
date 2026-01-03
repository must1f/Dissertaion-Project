#!/bin/bash

docker build -t jupyter .

docker run -p 8888:8888 jupyter