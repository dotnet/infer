using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;
using Microsoft.ML.Probabilistic.Collections;

namespace Microsoft.ML.Probabilistic.Tests.Core
{
    public class EnumerableExtensionsTests
    {
        [Fact]
        public void SumUInt_EmptyArray_ReturnsZero()
        {
            Assert.Equal(0U, new uint[0].Sum());
        }

        [Fact]
        public void SumULong_EmptyArray_ReturnsZero()
        {
            Assert.Equal(0UL, new ulong[0].Sum());
        }
    }
}
