#!/bin/sh
#enabling all the warnings using -Wall and -Wrxtra for extra warnings

#enable tracing for seeing whats going on
set -xe
clang -Wall -Wextra -o gates gates.c -lm