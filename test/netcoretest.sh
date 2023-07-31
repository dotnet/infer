#!/bin/bash

# Licensed to the .NET Foundation under one or more agreements.
# The .NET Foundation licenses this file to you under the MIT license.
# See the LICENSE file in the project root for more information.

# Script for testing Infer.Net on net core.
# Input parameters: build configuration ('Release' by default) 
# It saves results of testing in the `netcoretest-result*.xml` files
# in the working directory.

configuration=$1
if [ -z $configuration ]
then
    configuration=Release
fi

compath=/bin/${configuration}/net6.0/

# filter for parallel test run
#parallel_filter='-notrait Platform=x86 -notrait Category=OpenBug -notrait Category=BadTest -notrait Category=CompilerOptionsTest -notrait Category=CsoftModel -notrait Category=ModifiesGlobals -notrait Category=DistributedTest -notrait Category=Performance'
parallel_filter='--filter (Platform!=x86)&(Category!=OpenBug)&(Category!=BadTest)&(Category!=CompilerOptionsTest)&(Category!=CsoftModel)&(Category!=ModifiesGlobals)&(Category!=DistributedTest)&(Category!=Performance)'

# filter for sequential test run
#sequential_filter='-notrait Platform=x86 -trait Category=CsoftModel -trait Category=ModifiesGlobals -trait Category=DistributedTest -trait Category=Performance -notrait Category=OpenBug -notrait Category=BadTest -notrait Category=CompilerOptionsTest'
sequential_filter='--filter (Platform!=x86)&(Category!=OpenBug)&(Category!=BadTest)&(Category!=CompilerOptionsTest)&(Category=CsoftModel|Category=ModifiesGlobals|Category=DistributedTest|Category=Performance)'

exitcode=0

echo -e "\033[44;37m=====================PARALLEL TESTS RUNNING============================\033[0m"
cp "Tests/parallel.xunit.runner.json" "Tests/xunit.runner.json"
dotnet test ../Infer.sln -c "$configuration" $parallel_filter --logger "trx;logfilename=netcoretest-result.trx"
if [ 0 -ne $? ]
then
    echo -e "\033[5;41;1;37mParallel running failure!\033[0m"
    exitcode=1
else
    echo -e "\033[32;1mParallel running success!\033[0m"
fi

echo -e "\033[44;37m=====================SEQUENTIAL TESTS RUNNING=========================\033[0m"
# See https://xunit.github.io/docs/configuration-files
cp "Tests/sequential.xunit.runner.json" "Tests/xunit.runner.json"
dotnet test Tests -c "$configuration" $sequential_filter --logger "trx;logfilename=netcoretest-result2.trx"
if [ 0 -ne $? ]
then
    echo -e "\033[5;41;1;37mSequential running failure!\033[0m"
    exitcode=1
else
    echo -e "\033[32;1mSequential running success!\033[0m"
fi
cp "Tests/parallel.xunit.runner.json" "Tests/xunit.runner.json"

exit $exitcode
