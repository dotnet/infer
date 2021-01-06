// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Tests.TestAllCompilerOptions
{
    using Xunit;

    class Program
    {
        static void Main()
        {
            var failed = CompilerOptionsTestUtils.TestAllCompilerOptions(loadTestAssembly: true);
            Assert.Equal(0, failed);
        }
    }
}
