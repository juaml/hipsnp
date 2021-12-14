#!/bin/bash
mkdir tmp
cd tmp
git clone https://github.com/gavinband/qctool
cd qctool
./waf configure --prefix=/usr/bin/
./waf
./waf install