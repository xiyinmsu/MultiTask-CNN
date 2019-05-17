#!/usr/bin/env sh

./build/tools/caffe train -solver examples/MultiPIE/multitask/exp/solver.prototxt -snapshot examples/MultiPIE/multitask/exp/model/identity_exp_iter_1870875.solverstate  -gpu 5

