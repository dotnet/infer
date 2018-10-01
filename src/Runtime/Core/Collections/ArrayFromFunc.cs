// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Collections
{
    /// <summary>
    /// A virtual read-only 1D array whose elements are provided by a function.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class ArrayFromFunc<T> : IArray<T>
    {
        protected int count;
        protected Func<int, T> getItem;

        public ArrayFromFunc(int count, Func<int, T> getItem)
        {
            this.count = count;
            this.getItem = getItem;
        }

        public int IndexOf(T item)
        {
            throw new NotImplementedException();
        }

        public void Insert(int index, T item)
        {
            throw new NotImplementedException();
        }

        public void RemoveAt(int index)
        {
            throw new NotImplementedException();
        }

        public T this[int index]
        {
            get { return getItem(index); }
            set { throw new NotImplementedException(); }
        }

        public void Add(T item)
        {
            throw new NotImplementedException();
        }

        public void Clear()
        {
            throw new NotImplementedException();
        }

        public bool Contains(T item)
        {
            throw new NotImplementedException();
        }

        public void CopyTo(T[] array, int arrayIndex)
        {
            for (int i = 0; i < count; i++)
            {
                array[arrayIndex + i] = this[i];
            }
        }

        public int Count
        {
            get { return count; }
        }

        public bool IsReadOnly
        {
            get { return true; }
        }

        public bool Remove(T item)
        {
            throw new NotImplementedException();
        }

        public IEnumerator<T> GetEnumerator()
        {
            for (int i = 0; i < count; i++)
            {
                yield return this[i];
            }
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public int Rank
        {
            get { return 1; }
        }

        public int GetLength(int dimension)
        {
            if (dimension != 0) throw new IndexOutOfRangeException("requested dimension " + dimension + " of a 1D array.");
            return Count;
        }
    }

    /// <summary>
    /// A virtual read-only 2D array whose elements are provided by a function.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class ArrayFromFunc2D<T> : IArray2D<T>
    {
        public readonly int Length0, Length1;
        protected Func<int, int, T> getItem;

        public ArrayFromFunc2D(int length0, int length1, Func<int, int, T> getItem)
        {
            this.Length0 = length0;
            this.Length1 = length1;
            this.getItem = getItem;
        }

        public T this[int row, int column]
        {
            get { return getItem(row, column); }
            set { throw new NotImplementedException(); }
        }

        public int Rank
        {
            get { return 2; }
        }

        public int GetLength(int dimension)
        {
            if (dimension == 0) return Length0;
            else if (dimension == 1) return Length1;
            else throw new IndexOutOfRangeException("requested dimension " + dimension + " of a 2D array.");
        }

        public int IndexOf(T item)
        {
            throw new NotImplementedException();
        }

        public void Insert(int index, T item)
        {
            throw new NotImplementedException();
        }

        public void RemoveAt(int index)
        {
            throw new NotImplementedException();
        }

        public T this[int index]
        {
            get { throw new NotImplementedException(); }
            set { throw new NotImplementedException(); }
        }

        public void Add(T item)
        {
            throw new NotImplementedException();
        }

        public void Clear()
        {
            throw new NotImplementedException();
        }

        public bool Contains(T item)
        {
            throw new NotImplementedException();
        }

        public void CopyTo(T[] array, int arrayIndex)
        {
            throw new NotImplementedException();
        }

        public int Count
        {
            get { return Length0*Length1; }
        }

        public bool IsReadOnly
        {
            get { return false; }
        }

        public bool Remove(T item)
        {
            throw new NotImplementedException();
        }

        public IEnumerator<T> GetEnumerator()
        {
            for (int i = 0; i < Length0; i++)
            {
                for (int j = 0; j < Length1; j++)
                {
                    yield return this[i, j];
                }
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}