// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Utilities
{
    /// <summary>
    /// An equality comparer that compares arrays based on content.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class ArrayComparer<T> : IEqualityComparer<T[]>
    {
        private IEqualityComparer<T> elementComparer;

        public ArrayComparer()
        {
            elementComparer = EqualityComparer<T>.Default;
        }

        public ArrayComparer(IEqualityComparer<T> elementComparer)
        {
            this.elementComparer = elementComparer;
        }

        public bool Equals(T[] x, T[] y)
        {
            if (x.Length != y.Length) return false;
            for (int i = 0; i < x.Length; i++)
            {
                if (!elementComparer.Equals(x[i], y[i])) return false;
            }
            return true;
        }

        public int GetHashCode(T[] array)
        {
            int hash = Hash.Start;
            for (int i = 0; i < array.Length; i++)
            {
                hash = Hash.Combine(hash, elementComparer.GetHashCode(array[i]));
            }
            return hash;
        }
    }
}