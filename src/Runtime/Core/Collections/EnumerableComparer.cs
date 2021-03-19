// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Collections
{
    /// <summary>
    /// An equality comparer for IEnumerable that requires elements at the same position to match
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class EnumerableComparer<T> : IEqualityComparer<IEnumerable<T>>, IEqualityComparer<ICollection<T>>
    {
        public bool Equals(IEnumerable<T> x, IEnumerable<T> y)
        {
            return EnumerableExtensions.AreEqual(x, y);
        }

        public int GetHashCode(IEnumerable<T> obj)
        {
            return Hash.GetHashCodeAsSequence(obj);
        }

        public bool Equals(ICollection<T> x, ICollection<T> y)
        {
            return EnumerableExtensions.AreEqual(x, y);
        }

        public int GetHashCode(ICollection<T> obj)
        {
            return Hash.GetHashCodeAsSequence(obj);
        }
    }
}