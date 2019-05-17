#!/usr/bin/env sh

./build/tools/caffe train -solver examples/MultiPIE/multitask/illum/solver.prototxt -snapshot examples/MultiPIE/multitask/illum/model/identity_illum_iter_1870875.solverstate  -gpu 1

