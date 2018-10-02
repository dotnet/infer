#!/bin/bash

# Licensed to the .NET Foundation under one or more agreements.
# The .NET Foundation licenses this file to you under the MIT license.
# See the LICENSE file in the project root for more information.

# Script for testing Infer.Net on net core.
# Input parameters: build configuration ('Release' by default) 
# It saves results of testing in the `netcore-seqtests.xml` and `netcore-partests.xml` files
# in executing directory.

configuration=$1
if [ -z $configuration ]
then
    configuration=Release
fi

# files to save results
xml_parallel=netcore-partests.xml
xml_sequential=netcore-seqtests.xml


echo "Script for running xUnit tests."
echo -e "Usage: \033[33;40m ./netcoretest.sh [Release|Debug]\033[0m"
echo -e "Script stores info"
echo -e "    about parallel test running in            \033[35;40m${xml_parallel}\033[0m"
echo -e "    about sequential test running in          \033[35;40m${xml_sequential}\033[0m"

compath=/bin/${configuration}/netcoreapp2.0/

# Please note that order of test assemblies is important. *.Learners.Tests.dll must be first, otherwise BinaryFormatter-based tests will fail.
# These failures don't occur when testing *.Learners.Tests separately so this is probably xUnit issue.
dlls="Learners/LearnersTests${compath}Microsoft.ML.Probabilistic.Learners.Tests.dll Tests${compath}Microsoft.ML.Probabilistic.Tests.dll TestPublic${compath}TestPublic.dll"

# path to the xunit runner
runner=~/.nuget/packages/xunit.runner.console/2.3.1/tools/netcoreapp2.0/xunit.console.dll

# filter for parallel test run
parallel_filter='-notrait Platform=x86 -notrait Category=OpenBug -notrait Category=BadTest -notrait Category=CompilerOptionsTest -notrait Category=CsoftModel -notrait Category=ModifiesGlobals -notrait Category=DistributedTest -notrait Category=Performance'

# filter for sequential test run
sequential_filter='-notrait Platform=x86 -trait Category=CsoftModel -trait Category=ModifiesGlobals -trait Category=DistributedTests -trait Category=Performance -notrait Category=OpenBug -notrait Category=BadTest -notrait Category=CompilerOptionsTest'

exitcode=0

echo -e "\033[44;37m=====================PARALLEL TESTS RUNNING============================\033[0m"
dotnet "$runner" $dlls $parallel_filter -xml $xml_parallel
if [ 0 -ne $? ]
then
    echo -e "\033[5;41;1;37mParallel running failure!\033[0m"
    exitcode=1
else
    echo -e "\033[32;1mParallel running success!\033[0m"
fi

echo -e "\033[44;37m=====================SEQUENTIAL TESTS RUNNING=========================\033[0m"
dotnet "$runner" $dlls $sequential_filter -parallel none -xml $xml_sequential
if [ 0 -ne $? ]
then
    echo -e "\033[5;41;1;37mSequential running failure!\033[0m"
    exitcode=1
else
    echo -e "\033[32;1mSequential running success!\033[0m"
fi

exit $exitcode
