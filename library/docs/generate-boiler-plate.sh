#!/bin/bash

# Get the name of the library
codebase_name=$1
author_name=$2
version=$3

# Generate the boiler plate code
sphinx-quickstart --quiet -p $codebase_name -a $author_name -v $version
