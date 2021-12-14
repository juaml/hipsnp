#!/bin/bash
mkdir tmp
cd tmp || exit
# git clone https://github.com/gavinband/qctool.git
# cd qctool || exit
# ./waf configure --prefix=/usr/bin/
# ./waf
# ./waf install
wget https://www.well.ox.ac.uk/~gav/resources/qctool_v2.2.0-CentOS_Linux7.8.2003-x86_64.tgz
tar xzf qctool_v2.2.0-CentOS_Linux7.8.2003-x86_64.tgz
cd "qctool_v2.2.0-CentOS Linux7.8.2003-x86_64" || exit
cp qctool /usr/bin/
