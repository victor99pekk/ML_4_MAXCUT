# Get the directory where this Makefile lives
ROOT_DIR := $(CURDIR)
datatype ?= debug

# Default target (so 'make' runs this)
.PHONY: all
all: gen_data

# Generate data
.PHONY: gen_data
gen_data:
# make gen_data edge_mode=real datatyope=debug nbr_nodes=21
	python3 $(ROOT_DIR)/data/gen_maxcut_data.py --nbr_nodes $(nbr_nodes) --graph_type $(graph_type) --data_type $(data_type)
