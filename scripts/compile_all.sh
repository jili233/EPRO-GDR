#!/usr/bin/env bash

this_dir=$(dirname "$(realpath "$0")")

echo ""
echo "********build fps************"
cd $this_dir/../core/csrc/fps/
rm -rf build
python setup.py

echo ""
echo "********build cpp egl renderer************"
cd ../../../lib/egl_renderer/
rm -rf build/
python setup.py build_ext --inplace