#!/bin/bash
mkdir tmp
cd tmp
curl https://code.enkre.net/qctool/zip/release/qctool.tgz --output qctool.tgz
tar xzf qctool.tgz
cd qctool
./waf-1.5.18 configure
./waf-1.5.18
cp ./build/release/qctool_v2.0-release /usr/bin/qctool