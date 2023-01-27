#!/bin/bash
# Install tensorflow_gnn
tfgnn_validate_graph_schema --logtostderr \
    --graph_schema=example_schema.pbtxt

tfgnn_generate_training_data \
    --graph_schema=example_schema.pbtxt \
    --num_examples=3 \
    --examples=gnn-examples.tfrecord

tfgnn_print_training_data \
    --graph_schema=example_schema.pbtxt \
    --examples=gnn-examples.tfrecord \
    --mode=textproto > gnn-examples.txt
