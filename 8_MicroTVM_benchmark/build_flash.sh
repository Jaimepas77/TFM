#!/bin/bash

# This script is used to build and flash the project to the board

# Exit if project is found
if [ -d "project" ]; then
    echo "Project already exists"
    exit 1
fi
tar -xf project.tar
# Exit if project is not found
if [ ! -d "project" ]; then
    echo "Project not found"
    exit 1
fi
cd project
mkdir build
cd build
cmake ..
# Use the number of available CPU cores for parallel jobs
NUM_CORES=$(nproc)
make -j$((NUM_CORES - 1))
west flash
cd ../..
rm -rf project
