// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Collections
{
    /// <summary>
    /// An equality comparer for IList that requires elements at the same index to match
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class ReadOnlyListComparer<T> : IEqualityComparer<IReadOnlyList<T>>
    {
        public static ReadOnlyListComparer<T> Default { get; } = new ReadOnlyListComparer<T>();

        public bool Equals(IReadOnlyList<T> x, IReadOnlyList<T> y)
        {
            return ReadOnlyListComparer<T>.EqualLists(x, y);
        }

        int IEqualityComparer<IReadOnlyList<T>>.GetHashCode(IReadOnlyList<T> list)
        {
            return GetHashCode(list);
        }

        public static bool EqualLists(IReadOnlyList<T> x, IReadOnlyList<T> y)
        {
            if (x == null) return y == null;
            if (y == null) return false;
            if (x.Count != y.Count) return false;
            for (int i = 0; i < x.Count; i++) if (!Util.AreEqual(x[i], y[i])) return false;
            return true;
        }

        public static int GetHashCode(IReadOnlyList<T> list)
        {
            int hash = Hash.Combine(Hash.Start, list.Count);
            foreach (T obj in list)
            {
                hash = Hash.Combine(hash, obj == null ? 0 : obj.GetHashCode());
            }
            return hash;
        }
    }
}