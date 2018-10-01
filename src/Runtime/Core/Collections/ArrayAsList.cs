// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Collections
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Indicates if the object can convert to and from an array.
    /// </summary>
    /// <remarks>
    /// Possessing this interface implies that the object also has a constructor accepting an array type,
    /// such that <c>new T(this.ToArray())</c> is equivalent to <c>this.Clone()</c>.
    /// </remarks>
    public interface ConvertibleToArray
    {
        Array ToArray();
    }

    /// <summary>
    /// Wraps a multidimensional array to look like a linear list.
    /// </summary>
    /// <remarks>
    /// In the MSDN documentation, the Array class claims to implement IList.  In fact, this interface is only supported
    /// for one-dimensional arrays.  ArrayAsList provides this missing functionality for multidimensional arrays.
    /// </remarks>
    public class ArrayAsList<T> : IList<T>, ConvertibleToArray
    {
        private Array array;

        public Array Array
        {
            get { return array; }
        }

        private T[] array1D;

        public T[] Array1D
        {
            get { return array1D; }
        }

        private T[,] array2D;

        public T[,] Array2D
        {
            get { return array2D; }
        }

        protected int[] strides;

        public ArrayAsList(Array array)
        {
            this.array = array;
            if (array.Rank == 1) array1D = (T[]) array;
            else if (array.Rank == 2) array2D = (T[,]) array;
            else strides = StringUtil.ArrayStrides(StringUtil.ArrayDimensions(array));
        }

        public ArrayAsList(int length)
            : this(new T[length])
        {
        }

        public ArrayAsList(int length0, int length1)
            : this(new T[length0,length1])
        {
        }

        public ArrayAsList(params int[] lengths)
            : this(Array.CreateInstance(typeof (T), lengths))
        {
        }

        public Array ToArray()
        {
            return array;
        }

        public TRet ToArray<TRet>()
        {
            return (TRet) (object) array;
        }

        public int GetLength(int dimension)
        {
            return array.GetLength(dimension);
        }

        public int[] GetLengths()
        {
            return StringUtil.ArrayDimensions(array);
        }

        public int IndexOf(T item)
        {
            if (array1D != null) return Array.IndexOf(array1D, item);
            else return array.IndexOf<T>(item);
        }

        public void Insert(int index, T item)
        {
            throw new NotSupportedException();
        }

        public void RemoveAt(int index)
        {
            throw new NotSupportedException();
        }

        public T this[int index]
        {
            get
            {
                if (array1D != null) return array1D[index];
                else if (array2D != null)
                {
                    int cols = array2D.GetLength(1);
                    return array2D[index/cols, index%cols];
                }
                else
                {
                    int[] mIndex = new int[array.Rank];
                    StringUtil.LinearIndexToMultidimensionalIndex(index, strides, mIndex);
                    return (T) array.GetValue(mIndex);
                }
            }
            set
            {
                if (array1D != null) array1D[index] = value;
                else if (array2D != null)
                {
                    int cols = array2D.GetLength(1);
                    array2D[index/cols, index%cols] = value;
                }
                else
                {
                    int[] mIndex = new int[array.Rank];
                    StringUtil.LinearIndexToMultidimensionalIndex(index, strides, mIndex);
                    array.SetValue(value, mIndex);
                }
            }
        }

        public T this[int i, int j]
        {
            get
            {
                //return (T)array.GetValue(i,j);
                return array2D[i, j];
            }
            set
            {
                //array.SetValue(value,i,j);
                array2D[i, j] = value;
            }
        }

        public T this[params int[] index]
        {
            get { return (T) array.GetValue(index); }
            set { array.SetValue(value, index); }
        }

        public int FindIndex(Predicate<T> predicate)
        {
            if (array1D != null) return Array.FindIndex(array1D, predicate);
            else return array.FindIndex(predicate);
        }
#if false
        public void ForEachIndex(Action<int[]> action)
        {
            int[] mIndex = new int[array.Rank];
            for (int index = 0; index < Count; index++) {
                StringUtil.LinearIndexToMultidimensionalIndex(index, strides, mIndex);
                action(mIndex);
            }
        }
#endif

        public void ForEach(Action<int, T> action)
        {
            if (array1D != null)
            {
                for (int i = 0; i < array1D.Length; i++)
                {
                    action(i, array1D[i]);
                }
            }
            else array.ForEach(action);
        }

        public void ForEach(Action<T> action)
        {
            if (array1D != null)
            {
                for (int i = 0; i < array1D.Length; i++)
                {
                    action(array1D[i]);
                }
            }
            else array.ForEach<T>(action);
        }

        public void ForEach(Array a, Action<T, T> action)
        {
            ForEach<T>(a, action);
        }

        public void ForEach<U>(Array a, Action<T, U> action)
        {
            if (a.Length != this.Count) throw new ArgumentException("a.Length (" + a.Length + ") != this.Count (" + this.Count + ")");
            if (array1D != null)
            {
                U[] a1D = (U[]) a;
                for (int i = 0; i < array1D.Length; i++)
                {
                    action(array1D[i], a1D[i]);
                }
            }
            else if (array2D != null)
            {
                U[,] a2D = (U[,]) a;
                for (int i = 0; i < array2D.GetLength(0); i++)
                {
                    for (int j = 0; j < array2D.GetLength(1); j++)
                    {
                        action(array2D[i, j], a2D[i, j]);
                    }
                }
            }
            else
            {
                int[] mIndex = new int[array.Rank];
                for (int index = 0; index < Count; index++)
                {
                    StringUtil.LinearIndexToMultidimensionalIndex(index, strides, mIndex);
                    T value = (T) array.GetValue(mIndex);
                    U aValue = (U) a.GetValue(mIndex);
                    action(value, aValue);
                }
            }
        }

        public void ForEach(Array a, Array b, Action<T, T, T> action)
        {
            if (a.Length != this.Count) throw new ArgumentException("a.Length (" + a.Length + ") != this.Count (" + this.Count + ")");
            if (b.Length != this.Count) throw new ArgumentException("b.Length (" + b.Length + ") != this.Count (" + this.Count + ")");
            if (array1D != null)
            {
                T[] a1D = (T[]) a;
                T[] b1D = (T[]) b;
                for (int i = 0; i < array1D.Length; i++)
                {
                    action(array1D[i], a1D[i], b1D[i]);
                }
            }
            else if (array2D != null)
            {
                T[,] a2D = (T[,]) a;
                T[,] b2D = (T[,]) b;
                for (int i = 0; i < array2D.GetLength(0); i++)
                {
                    for (int j = 0; j < array2D.GetLength(1); j++)
                    {
                        action(array2D[i, j], a2D[i, j], b2D[i, j]);
                    }
                }
            }
            else
            {
                int[] mIndex = new int[array.Rank];
                for (int index = 0; index < Count; index++)
                {
                    StringUtil.LinearIndexToMultidimensionalIndex(index, strides, mIndex);
                    T value = (T) array.GetValue(mIndex);
                    T aValue = (T) a.GetValue(mIndex);
                    T bValue = (T) b.GetValue(mIndex);
                    action(value, aValue, bValue);
                }
            }
        }

        public void ModifyAll(Func<T, T> converter)
        {
            if (array1D != null)
            {
                for (int i = 0; i < array1D.Length; i++)
                {
                    array1D[i] = converter(array1D[i]);
                }
            }
            else if (array2D != null)
            {
                for (int i = 0; i < array2D.GetLength(0); i++)
                    for (int j = 0; j < array2D.GetLength(1); j++)
                        array2D[i, j] = converter(array2D[i, j]);
            }
            else
            {
                int[] mIndex = new int[array.Rank];
                for (int index = 0; index < Count; index++)
                {
                    StringUtil.LinearIndexToMultidimensionalIndex(index, strides, mIndex);
                    T value = (T) array.GetValue(mIndex);
                    value = converter(value);
                    array.SetValue(value, mIndex);
                }
            }
        }

        public void ModifyAll(Array a, Func<T, T, T> converter)
        {
            if (a.Length != this.Count) throw new ArgumentException("a.Length (" + a.Length + ") != this.Count (" + this.Count + ")");
            if (array1D != null)
            {
                T[] a1D = (T[]) a;
                for (int i = 0; i < array1D.Length; i++) array1D[i] = converter(array1D[i], a1D[i]);
            }
            else if (array2D != null)
            {
                T[,] a2D = (T[,]) a;
                for (int i = 0; i < array2D.GetLength(0); i++)
                    for (int j = 0; j < array2D.GetLength(1); j++)
                        array2D[i, j] = converter(array2D[i, j], a2D[i, j]);
            }
            else
            {
                int[] mIndex = new int[array.Rank];
                for (int index = 0; index < Count; index++)
                {
                    StringUtil.LinearIndexToMultidimensionalIndex(index, strides, mIndex);
                    T value = (T) array.GetValue(mIndex);
                    T aValue = (T) a.GetValue(mIndex);
                    value = converter(value, aValue);
                    array.SetValue(value, mIndex);
                }
            }
        }

        public void ModifyAll(Array a, Array b, Func<T, T, T, T> converter)
        {
            if (a.Length != this.Count) throw new ArgumentException("a.Length (" + a.Length + ") != this.Count (" + this.Count + ")");
            if (b.Length != this.Count) throw new ArgumentException("b.Length (" + b.Length + ") != this.Count (" + this.Count + ")");
            if (array1D != null)
            {
                T[] a1D = (T[]) a;
                T[] b1D = (T[]) b;
                for (int i = 0; i < array1D.Length; i++) array1D[i] = converter(array1D[i], a1D[i], b1D[i]);
            }
            else if (array2D != null)
            {
                T[,] a2D = (T[,]) a;
                T[,] b2D = (T[,]) b;
                for (int i = 0; i < array2D.GetLength(0); i++)
                    for (int j = 0; j < array2D.GetLength(1); j++)
                        array2D[i, j] = converter(array2D[i, j], a2D[i, j], b2D[i, j]);
            }
            else
            {
                int[] mIndex = new int[array.Rank];
                for (int index = 0; index < Count; index++)
                {
                    StringUtil.LinearIndexToMultidimensionalIndex(index, strides, mIndex);
                    T value = (T) array.GetValue(mIndex);
                    T aValue = (T) a.GetValue(mIndex);
                    T bValue = (T) b.GetValue(mIndex);
                    value = converter(value, aValue, bValue);
                    array.SetValue(value, mIndex);
                }
            }
        }

        public void Add(T item)
        {
            throw new NotSupportedException();
        }

        public void Clear()
        {
            Array.Clear(array, 0, array.Length);
        }

        public bool Contains(T item)
        {
            return (IndexOf(item) != -1);
        }

        public void CopyTo(T[] array, int arrayIndex)
        {
            //Array.Copy(this.array,0,array,arrayIndex,this.array.Length);
            ForEach(delegate(int i, T value) { array[i + arrayIndex] = value; });
        }

        public int Count
        {
            get { return array.Length; }
        }

        public bool IsReadOnly
        {
            get { return array.IsReadOnly; }
        }

        public bool Remove(T item)
        {
            throw new NotSupportedException();
        }

        public IEnumerator<T> GetEnumerator()
        {
            foreach (T item in array)
            {
                yield return item;
            }
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return array.GetEnumerator();
        }

        public override string ToString()
        {
            return StringUtil.ArrayToString(array);
        }

        public override bool Equals(object obj)
        {
            if (!(obj is ArrayAsList<T>)) return false;
            ArrayAsList<T> that = (ArrayAsList<T>) obj;
            bool equal = true;
            ForEach<T>(that.array,
                       delegate(T item, T bItem) { if (!item.Equals(bItem)) equal = false; });
            return equal;
        }

        public override int GetHashCode()
        {
            int hash = Hash.Start;
            for (int dimension = 0; dimension < array.Rank; dimension++)
            {
                hash = Hash.Combine(hash, GetLength(dimension));
            }
            ForEach(delegate(T item) { hash = Hash.Combine(hash, item.GetHashCode()); });
            return hash;
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}