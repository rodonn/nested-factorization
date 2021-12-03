#!/bin/bash
g++ -std=c++11 -Wall -o hpf hpf.cpp `gsl-config --cflags --libs`

