// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Xunit;

namespace Microsoft.ML.Probabilistic.Tests
{
    
    public class CompilerOptionsTest
    {
        [Fact]
        [Trait("Category", "CompilerOptionsTest")]
        public void TestAllCompilerOptions()
        {
            var failed = TestUtils.TestAllCompilerOptions();
            Assert.Equal(0, failed);
        }
    }
}
