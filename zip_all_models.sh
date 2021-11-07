#!/bin/sh

set -ex
export COPYFILE_DISABLE=1

base_path=models
for directory in `ls $base_path`; do
    echo "Zip $base_path/$directory"
    pushd "$base_path/$directory"
    filename=$(basename `pwd`).tar.gz
    tar -zcvf "$filename" .
    mv $filename ../..
    popd
done
