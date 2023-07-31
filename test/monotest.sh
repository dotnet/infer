#!/bin/bash

# Licensed to the .NET Foundation under one or more agreements.
# The .NET Foundation licenses this file to you under the MIT license.
# See the LICENSE file in the project root for more information.

# Script for testing Infer.Net on Mono.
# Input parameters: test dll list.
# It saves results of testing in the `parallel.html` and `sequential.html` files
# in executing directory.

if [ -z $1 ]
then
    echo -e "\033[5;41;1;37mEmpty dll list!\033[0m"
    echo -e "Give\033[33;40m -h \033[0mor\033[33;40m --help \033[0margument to get reference"
    exit 1
fi

if [ $1 = "-h" -o $1 = "--help" ]
then
    echo "Script for running xUnit tests."
    echo -e "Usage: \033[33;40m ./monotest.sh <dll list>\033[0m"
    echo -e "Script stores info"
    echo -e "    about parallel test running in            \033[35;40mparallel.html\033[0m"
    echo -e "    about sequential test running in          \033[35;40msequential.html\033[0m"
    exit 0
fi

# files to save results
html_parallel=parallel.html
html_sequential=sequential.html

# path to the xunit runner
runner=~/.nuget/packages/xunit.runner.console/2.4.1/tools/net452/xunit.console.exe

# filter for parallel test run
parallel_filter='-notrait Platform=x86 -notrait Category=OpenBug -notrait Category=BadTest -notrait Category=CompilerOptionsTest -notrait Category=CsoftModel -notrait Category=ModifiesGlobals -notrait Category=DistributedTest -notrait Category=Performance'

# filter for sequential test run
sequential_filter='-notrait Platform=x86 -trait Category=CsoftModel -trait Category=ModifiesGlobals -trait Category=DistributedTest -trait Category=Performance -notrait Category=OpenBug -notrait Category=BadTest -notrait Category=CompilerOptionsTest'

exitcode=0

echo -e "\033[44;37m=====================PARALLEL TESTS RUNNING============================\033[0m"
mono "$runner" $@ $parallel_filter -html $html_parallel
if [ 0 -ne $? ]
then
    echo -e "\033[5;41;1;37mParallel running failure!\033[0m"
    exitcode=1
else
    echo -e "\033[32;1mParallel running success!\033[0m"
fi

echo -e "\033[44;37m=====================SEQUENTIAL TESTS RUNNING=========================\033[0m"
mono "$runner" $@ $sequential_filter -parallel none -html $html_sequential
if [ 0 -ne $? ]
then
    echo -e "\033[5;41;1;37mSequential running failure!\033[0m"
    exitcode=1
else
    echo -e "\033[32;1mSequential running success!\033[0m"
fi

exit $exitcode
