// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Diagnostics;
using Xunit;

namespace Microsoft.ML.Probabilistic.Tests
{
    public class PlatformTests
    {
        [Fact]
        [Trait("Platform","x64")]
        public void Is64BitProcessTest()
        {
            Assert.True(System.Environment.Is64BitProcess);
            Trace.WriteLine("Microsoft.ML.Probabilistic.Tests are running as 64 bit process");
        }

        [Fact]
        [Trait("Platform", "x86")]
        public void Is32BitProcessTest()
        {
            Assert.False(System.Environment.Is64BitProcess);
            Trace.WriteLine("Microsoft.ML.Probabilistic.Tests are running as 32 bit process");
        }
    }
}

