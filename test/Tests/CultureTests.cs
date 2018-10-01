// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Xunit;
using System.Globalization;
using System.Threading;

namespace Microsoft.ML.Probabilistic.Tests
{
    
    public class CultureTests
    {
        [Fact]
        public void TurkeyTest()
        {
            var originalCulture = Thread.CurrentThread.CurrentCulture;
            var originalUICulture = Thread.CurrentThread.CurrentUICulture;
            try
            {
                var turkish = new CultureInfo("tr-TR");
                Thread.CurrentThread.CurrentCulture = turkish;
                Thread.CurrentThread.CurrentUICulture = turkish;
                new GateModelTests().MurphySprinklerTest();
            }
            finally
            {
                Thread.CurrentThread.CurrentCulture = originalCulture;
                Thread.CurrentThread.CurrentUICulture = originalUICulture;
            }
        }
    }
}
