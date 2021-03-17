// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
namespace Microsoft.ML.Probabilistic.Tests.Core
{
    using Xunit;
    using System.Collections.Generic;
    using System.Linq;

    
    public class UtilsTests
    {
        [Fact]
        public void GetElementType()
        {
            int rank;
            Assert.Equal(typeof(bool), Utilities.Util.GetElementType(typeof(IReadOnlyList<bool>), out rank));
            Assert.Equal(1, rank);
        }

        [Fact]
        public void TestCartesianProduct()
        {
            var lists = new List<IEnumerable<string>>
            {
                new List<string> {"a1", "a2", "a3" },
                new List<string> {"b1", "b2" },
                new List<string> {"c1", "c2" }
            };

            var cartProd = new HashSet<string>(Utilities.Util.CartesianProduct(lists).
                Select(x =>
                {
                    var arr = x.ToArray();
                    return $"{arr[0]}{arr[1]}{arr[2]}";
                }));

            Assert.Equal(3 * 2 * 2, cartProd.Count);
            Assert.Contains("a1b1c1", cartProd);
            Assert.Contains("a1b1c2", cartProd);
            Assert.Contains("a1b2c1", cartProd);
            Assert.Contains("a1b2c2", cartProd);
            Assert.Contains("a2b1c1", cartProd);
            Assert.Contains("a2b1c2", cartProd);
            Assert.Contains("a2b2c1", cartProd);
            Assert.Contains("a2b2c2", cartProd);
            Assert.Contains("a3b1c1", cartProd);
            Assert.Contains("a3b1c2", cartProd);
            Assert.Contains("a3b2c1", cartProd);
            Assert.Contains("a3b2c2", cartProd);
        }
    }
}
