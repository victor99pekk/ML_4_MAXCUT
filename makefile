# Get the directory where this Makefile lives
ROOT_DIR := $(CURDIR)
datatype ?= debug

# Default target (so 'make' runs this)
.PHONY: all
all: gen_data

# Generate data
.PHONY: gen_data
gen_data:
# make gen_data datatyope=debug nbr_nodes=21
	python3 $(ROOT_DIR)/data/gen_maxcut_data.py --nbr_nodes $(nbr_nodes) --graph_type $(graph_type) --data_type $(data_type)

train:
#!python neural_network/train_network.py
	python3 $(ROOT_DIR)/neural_network/train_network.py --nbr_nodes $(nbr_nodes) --model_name $(model_name)

evaluate:
	python3 $(ROOT_DIR)/neural_network/evaluate_network.py --n $(n) --model_name $(model_name) --compile_model $(compile_model)