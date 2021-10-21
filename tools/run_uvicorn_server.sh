#!/bin/bash

uvicorn inference_server:app --host 0.0.0.0 --reload --port 8080

