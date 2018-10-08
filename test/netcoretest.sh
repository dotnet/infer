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

compath=/bin/${configuration}/netcoreapp2.1/
dlls="Learners/LearnersTests${compath}Microsoft.ML.Probabilistic.Learners.Tests.dll Tests${compath}Microsoft.ML.Probabilistic.Tests.dll TestPublic${compath}TestPublic.dll"

# path to the xunit runner
runner=~/.nuget/packages/xunit.runner.console/2.3.1/tools/netcoreapp2.0/xunit.console.dll

# filter for parallel test run
parallel_filter='-notrait Platform=x86 -notrait Category=OpenBug -notrait Category=BadTest -notrait Category=CompilerOptionsTest -notrait Category=CsoftModel -notrait Category=ModifiesGlobals -notrait Category=DistributedTest -notrait Category=Performance'

# filter for sequential test run
sequential_filter='-notrait Platform=x86 -trait Category=CsoftModel -trait Category=ModifiesGlobals -trait Category=DistributedTests -trait Category=Performance -notrait Category=OpenBug -notrait Category=BadTest -notrait Category=CompilerOptionsTest'

exitcode=0
index=0

echo -e "\033[44;37m=====================PARALLEL TESTS RUNNING============================\033[0m"
for assembly in Learners/LearnersTests${compath}Microsoft.ML.Probabilistic.Learners.Tests.dll Tests${compath}Microsoft.ML.Probabilistic.Tests.dll TestPublic${compath}TestPublic.dll
do
    # Please note that due to xUnit issue we need to run tests for each assembly separately
    dotnet "$runner" "$assembly" $parallel_filter -xml "netcoretest-result${index}.xml"
    if [ 0 -ne $? ]
    then
        echo -e "\033[5;41;1;37mParallel running failure!\033[0m"
        exitcode=1
    else
        echo -e "\033[32;1mParallel running success!\033[0m"
    fi
    (( index++ ))
done

echo -e "\033[44;37m=====================SEQUENTIAL TESTS RUNNING=========================\033[0m"
dotnet "$runner" "Tests${compath}Microsoft.ML.Probabilistic.Tests.dll" $sequential_filter -parallel none -xml "netcoretest-result${index}.xml"
if [ 0 -ne $? ]
then
    echo -e "\033[5;41;1;37mSequential running failure!\033[0m"
    exitcode=1
else
    echo -e "\033[32;1mSequential running success!\033[0m"
fi

exit $exitcode
