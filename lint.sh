#!/bin/bash

echo "Linting package..."
pylint ./sim_tools

echo "Linting tests..."
pylint ./tests

echo "Linting notebooks..."
nbqa pylint ./docs