:: Licensed to the .NET Foundation under one or more agreements.
:: The .NET Foundation licenses this file to you under the MIT license.
:: See the LICENSE file in the project root for more information.
@echo off

set JDK_BIN_DIR="C:\Program Files\Java\jdk1.7.0_25\bin"
if not exist %JDK_BIN_DIR% goto :NO_JDK

set JAVAC=%JDK_BIN_DIR%\javac.exe
set JAR=%JDK_BIN_DIR%\jar.exe
set CLASS_FILES=MahoutRunner.class MahoutRunner$EntitySkipper.class MahoutRunner$RelatedEntityFinder.class MahoutRunner$RelatedItemFinder.class MahoutRunner$RelatedUserFinder.class

%JAVAC% -cp .\mahout-core-0.8-job.jar .\MahoutRunner.java  || goto :ERROR
%JAR% -cf MahoutRunner.jar %CLASS_FILES%  || goto :ERROR
del %CLASS_FILES% || goto :ERROR

goto :EOF

:ERROR
echo.
echo Interrupted due to error(s).
exit /b 1

:NO_JDK
echo JDK hasn't been found at %JDK_BIN_DIR%. Please specify the correct location.
exit /b 2