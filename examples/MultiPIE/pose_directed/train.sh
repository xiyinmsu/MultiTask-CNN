#!/usr/bin/env sh

./build/tools/caffe train -solver examples/MultiPIE/pose/solver.prototxt -snapshot examples/MultiPIE/pose/model/pose_iter_1870875.solverstate -gpu 5


