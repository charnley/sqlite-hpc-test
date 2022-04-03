
all: env

# Python

env:
	mamba env create -f ./environment.yml -p ./env --quiet
