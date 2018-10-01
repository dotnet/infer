# Licensed to the .NET Foundation under one or more agreements.
# The .NET Foundation licenses this file to you under the MIT license.
# See the LICENSE file in the project root for more information.

if [ $# -lt 1 ]; then
    echo Usage: $0 "<version>"
    exit 1
fi

for f in SharedAssemblyFileVersion.cs SharedAssemblyFileVersion.fs
do
    sed -i "s/\(Assembly\(File\)\?Version(\"\)[0-9]\+.[0-9]\+.[0-9]\+.[0-9]\+/\1$1/" ../src/Shared/$f
done