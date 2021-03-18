// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Collections
{
    /// <summary>
    /// An equality comparer that requires the sequence of keys and values to match
    /// </summary>
    /// <typeparam name="TKey"></typeparam>
    /// <typeparam name="TValue"></typeparam>
    public class DictionaryComparer<TKey, TValue> : EnumerableComparer<KeyValuePair<TKey, TValue>>, IEqualityComparer<Dictionary<TKey, TValue>>
    {
        public bool Equals(Dictionary<TKey, TValue> x, Dictionary<TKey, TValue> y)
        {
            return Equals((IEnumerable<KeyValuePair<TKey, TValue>>) x, (IEnumerable<KeyValuePair<TKey, TValue>>) y);
        }

        public int GetHashCode(Dictionary<TKey, TValue> obj)
        {
            return GetHashCode((IEnumerable<KeyValuePair<TKey, TValue>>) obj);
        }
    }
}