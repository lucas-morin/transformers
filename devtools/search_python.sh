#!/bin/bash

search=$1

find . \
  -type d \( -path ./venv -o -path ./venv_3.8 -o -path ./venv_3.9 -o -path ./venv_3.10 -o -path ./venv_3.11 \) \
  -prune \
  -o -name '*.py' \
  -exec grep -irn "${search}" {} + 
