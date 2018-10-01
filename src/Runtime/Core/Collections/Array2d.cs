// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Runtime.Serialization;

using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Collections
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// The base class for arrays of any rank using value equality.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    [Serializable]
    [DataContract]
    public abstract class ArrayBase<T> : IArray<T>, CanSetAllElementsTo<T>, IReadOnlyList<T>
    {
        [DataMember]
        protected internal T[] array;

        public abstract int Rank { get; }
        public abstract int GetLength(int dimension);

        public T this[int index]
        {
            get { return array[index]; }
            set { array[index] = value; }
        }

        public int IndexOf(T item)
        {
            return Array.IndexOf(array, item);
        }

        public void Insert(int index, T item)
        {
            throw new NotSupportedException();
        }

        public void RemoveAt(int index)
        {
            throw new NotSupportedException();
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

        public virtual void CopyTo(T[] array, int arrayIndex)
        {
            Array.Copy(this.array, 0, array, arrayIndex, this.array.Length);
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

        protected void CheckCompatible(IArray<T> that)
        {
            if (that.Rank != this.Rank) throw new ArgumentException("that.Rank (" + that.Rank + ") != this.Rank (" + this.Rank + ")");
            for (int dimension = 0; dimension < Rank; dimension++)
            {
                if (that.GetLength(dimension) != this.GetLength(dimension))
                {
                    throw new ArgumentException("that.GetLength(" + dimension + ") (" + that.GetLength(dimension) + ") != this.GetLength(" + dimension + ") (" +
                                                this.GetLength(dimension) + ")");
                }
            }
        }

        public override bool Equals(object obj)
        {
            if (!(obj is ArrayBase<T>)) return false;
            ArrayBase<T> that = (ArrayBase<T>) obj;
            for (int dimension = 0; dimension < Rank; dimension++)
            {
                if (that.GetLength(dimension) != this.GetLength(dimension)) return false;
            }
            for (int i = 0; i < array.Length; i++)
            {
                T item = array[i];
                T thatItem = that[i];
                if (!item.Equals(thatItem)) return false;
            }
            return true;
        }

        public override int GetHashCode()
        {
            int hash = Hash.Start;
            for (int dimension = 0; dimension < Rank; dimension++)
            {
                hash = Hash.Combine(hash, GetLength(dimension));
            }
            Array.ForEach(array, delegate(T item) { hash = Hash.Combine(hash, item.GetHashCode()); });
            return hash;
        }

        public virtual void SetAllElementsTo(T value)
        {
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = value;
            }
        }

        public int FindIndex(Predicate<T> predicate)
        {
            return Array.FindIndex(array, predicate);
        }

        public void ForEach(Action<T> action)
        {
            for (int i = 0; i < array.Length; i++)
            {
                action(array[i]);
            }
        }

        public void ForEach<U>(U[] array, Action<T, U> action)
        {
            if (array.Length != this.array.Length) throw new ArgumentException("array.Length (" + array.Length + ") != this.Count (" + this.Count + ")");
            for (int i = 0; i < array.Length; i++)
            {
                action(this.array[i], array[i]);
            }
        }

        public void ForEach<U, V>(U[] array, V[] array2, Action<T, U, V> action)
        {
            if (array.Length != this.array.Length) throw new ArgumentException("array.Length (" + array.Length + ") != this.Count (" + this.Count + ")");
            if (array2.Length != this.array.Length) throw new ArgumentException("array2.Length (" + array2.Length + ") != this.Count (" + this.Count + ")");
            for (int i = 0; i < this.array.Length; i++)
            {
                action(this.array[i], array[i], array2[i]);
            }
        }

        public void SetToFunction<U>(U[] array, Func<U, T> converter)
        {
            if (array.Length != this.array.Length) throw new ArgumentException("array.Length (" + array.Length + ") != this.Count (" + this.Count + ")");
            for (int i = 0; i < array.Length; i++)
                this.array[i] = converter(array[i]);
        }

        public void SetToFunction<U>(ArrayBase<U> array, Func<U, T> converter)
        {
            if (array.Count != this.array.Length) throw new ArgumentException("array.Count (" + array.Count + ") != this.Count (" + this.Count + ")");
            for (int i = 0; i < array.Count; i++)
                this.array[i] = converter(array[i]);
        }

        public void SetToFunction<U, V>(ArrayBase<U> array, V[] array2, Func<U, V, T> converter)
        {
            if (array.Count != this.array.Length) throw new ArgumentException("array.Count (" + array.Count + ") != this.Count (" + this.Count + ")");
            if (array2.Length != this.array.Length) throw new ArgumentException("array2.Length (" + array2.Length + ") != this.Count (" + this.Count + ")");
            for (int i = 0; i < array.Count; i++)
            {
                this.array[i] = converter(array[i], array2[i]);
            }
        }

        public void SetToFunction<U, V, W>(ArrayBase<U> array, V[] array2, W[] array3, Func<U, V, W, T> converter)
        {
            if (array.Count != this.array.Length) throw new ArgumentException("array.Count (" + array.Count + ") != this.Count (" + this.Count + ")");
            if (array2.Length != this.array.Length) throw new ArgumentException("array2.Length (" + array2.Length + ") != this.Count (" + this.Count + ")");
            if (array3.Length != this.array.Length) throw new ArgumentException("array3.Length (" + array3.Length + ") != this.Count (" + this.Count + ")");
            for (int i = 0; i < array.Count; i++)
            {
                this.array[i] = converter(array[i], array2[i], array3[i]);
            }
        }
    }

    /// <summary>
    /// A one-dimensional array with value equality.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    [Serializable]
    [DataContract]
    public class Array<T> : ArrayBase<T>, SettableTo<T[]>, SettableTo<Array<T>>, ICloneable, ConvertibleToArray
    {
        /// <summary>
        /// Parameterless constructor needed for serialization
        /// </summary>
        protected Array()
        {
        }

        public Array(int length)
        {
            array = new T[length];
        }

        public Array(T[] array)
            : this(array.Length)
        {
            SetTo(array);
        }

        public virtual void SetTo(T[] array)
        {
            if (array.Length != this.Count) throw new ArgumentException("array.Length (" + array.Length + ") != this.Count (" + this.Count + ")");
            array.CopyTo(this.array, 0);
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that"></param>
        public Array(Array<T> that)
            : this(that.Count)
        {
            SetTo(that);
        }

        public void SetTo(Array<T> that)
        {
            if (that.Count != Count) throw new ArgumentException("that.Count (" + that.Count + ") != this.Count (" + this.Count + ")");
            that.CopyTo(array, 0);
        }

        /// <summary>
        /// Clone the array but not the items in the array.
        /// </summary>
        /// <returns></returns>
        public virtual object Clone()
        {
            return new Array<T>(this);
        }

        public virtual void CopyTo(T[] array)
        {
            if (array.Length != this.Count) throw new ArgumentException("array.Length (" + array.Length + ") != this.Count (" + this.Count + ")");
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = this[i];
            }
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning disable 162
#endif

        public T[] ToArray()
        {
            if (true)
            {
                return array;
            }
            else
            {
                T[] result = new T[Count];
                CopyTo(result);
                return result;
            }
        }

#if SUPPRESS_UNREACHABLE_CODE_WARNINGS
#pragma warning restore 162
#endif

        Array ConvertibleToArray.ToArray()
        {
            return ToArray();
        }

        public static TOutput[] ToArray<TOutput>(Converter<T, TOutput> itemConverter, Array<T> array)
        {
            TOutput[] result = new TOutput[array.Count];
            for (int i = 0; i < array.Count; i++)
            {
                result[i] = itemConverter(array[i]);
            }
            return result;
        }

        public override int Rank
        {
            get { return 1; }
        }

        public override int GetLength(int dimension)
        {
            if (dimension == 0) return Count;
            else throw new IndexOutOfRangeException("requested dimension " + dimension + " of a 1D array.");
        }

#if false
        public T this[params int[] indices]
        {
            get
            {
                if (indices.Length != 1) throw new ArgumentException("provided " + indices.Length + " indices to a 1D array.");
                return this[indices[0]];
            }
            set
            {
                if (indices.Length != 1) throw new ArgumentException("provided " + indices.Length + " indices to a 1D array.");
                this[indices[0]] = value;
            }
        }
#endif

        public override bool Equals(object obj)
        {
            return (obj is Array<T>) && base.Equals(obj);
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }
    }

    /// <summary>
    /// A two-dimensional array with value equality.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    [Serializable]
    [DataContract]
    public class Array2D<T> : ArrayBase<T>, IArray2D<T>, SettableTo<T[,]>, SettableTo<Array2D<T>>, ICloneable, ConvertibleToArray
    {
        public readonly int Length0, Length1;

        /// <summary>
        /// Parameterless constructor needed for serialization
        /// </summary>
        protected Array2D()
        {
        }

        public Array2D(int Length0, int Length1)
            : this(Length0, Length1, new T[Length0*Length1])
        {
        }

        protected Array2D(int Length0, int Length1, T[] valuesRef)
        {
            this.Length0 = Length0;
            this.Length1 = Length1;
            array = valuesRef;
        }

        public Array2D(T[,] array)
            : this(array.GetLength(0), array.GetLength(1))
        {
            SetTo(array);
        }

        public virtual void SetTo(T[,] array)
        {
            if (array.GetLength(0) != Length0) throw new ArgumentException("array.GetLength(0) (" + array.GetLength(0) + ") != this.Length0 (" + this.Length0 + ")");
            if (array.GetLength(1) != Length1) throw new ArgumentException("array.GetLength(1) (" + array.GetLength(1) + ") != this.Length1 (" + this.Length1 + ")");
            for (int i = 0; i < Length0; i++)
            {
                for (int j = 0; j < Length1; j++)
                {
                    this[i, j] = array[i, j];
                }
            }
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="that"></param>
        public Array2D(Array2D<T> that)
            : this(that.Length0, that.Length1)
        {
            SetTo(that);
        }

        public void SetTo(Array2D<T> that)
        {
            if (that.Length0 != Length0) throw new ArgumentException("that.Length0 (" + that.Length0 + ") != this.Length0 (" + this.Length0 + ")");
            if (that.Length1 != Length1) throw new ArgumentException("that.Length1 (" + that.Length1 + ") != this.Length1 (" + this.Length1 + ")");
            that.CopyTo(array, 0);
        }

        /// <summary>
        /// Clone the array but not the items in the array.
        /// </summary>
        /// <returns></returns>
        public virtual object Clone()
        {
            return new Array2D<T>(this);
        }

        public virtual void CopyTo(T[,] array)
        {
            if (array.GetLength(0) != Length0) throw new ArgumentException("array.GetLength(0) (" + array.GetLength(0) + ") != this.Length0 (" + this.Length0 + ")");
            if (array.GetLength(1) != Length1) throw new ArgumentException("array.GetLength(1) (" + array.GetLength(1) + ") != this.Length1 (" + this.Length1 + ")");
            for (int i = 0; i < Length0; i++)
            {
                for (int j = 0; j < Length1; j++)
                {
                    array[i, j] = this[i, j];
                }
            }
        }

        public T[,] ToArray()
        {
            T[,] result = new T[Length0,Length1];
            CopyTo(result);
            return result;
        }

        Array ConvertibleToArray.ToArray()
        {
            return ToArray();
        }

        public static TOutput[,] ToArray<TOutput>(Converter<T, TOutput> itemConverter, Array2D<T> array)
        {
            TOutput[,] result = new TOutput[array.Length0,array.Length1];
            for (int i = 0; i < array.Length0; i++)
            {
                for (int j = 0; j < array.Length1; j++)
                {
                    result[i, j] = itemConverter(array[i, j]);
                }
            }
            return result;
        }

        public override string ToString()
        {
            int[] strides = new int[2];
            strides[0] = Length1;
            strides[1] = 1;
            int[] lowerBounds = new int[2];
            return StringUtil.ArrayToString(array, strides, lowerBounds);
        }

        public T this[int row, int column]
        {
            get { return array[row*Length1 + column]; }
            set { array[row*Length1 + column] = value; }
        }

        public override int Rank
        {
            get { return 2; }
        }

        public override int GetLength(int dimension)
        {
            if (dimension == 0) return Length0;
            else if (dimension == 1) return Length1;
            else throw new IndexOutOfRangeException("requested dimension " + dimension + " of a 2D array.");
        }

#if false
        public T this[params int[] indices]
        {
            get
            {
                if (indices.Length != 2) throw new ArgumentException("provided " + indices.Length + " indices to a 2D array.");
                return this[indices[0], indices[1]];
            }
            set
            {
                if (indices.Length != 2) throw new ArgumentException("provided " + indices.Length + " indices to a 2D array.");
                this[indices[0], indices[1]] = value;
            }
        }
#endif

        public override bool Equals(object obj)
        {
            return (obj is Array2D<T>) && base.Equals(obj);
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        public void ForEach<U>(U[,] array, Action<T, U> action)
        {
            if (array.GetLength(0) != Length0) throw new ArgumentException("array.GetLength(0) (" + array.GetLength(0) + ") != this.Length0 (" + this.Length0 + ")");
            if (array.GetLength(1) != Length1) throw new ArgumentException("array.GetLength(1) (" + array.GetLength(1) + ") != this.Length1 (" + this.Length1 + ")");
            for (int i = 0; i < Length0; i++)
            {
                for (int j = 0; j < Length1; j++)
                {
                    action(this[i, j], array[i, j]);
                }
            }
        }

        public void ForEach<U, V>(U[,] array, V[,] array2, Action<T, U, V> action)
        {
            if (array.GetLength(0) != Length0) throw new ArgumentException("array.GetLength(0) (" + array.GetLength(0) + ") != this.Length0 (" + this.Length0 + ")");
            if (array.GetLength(1) != Length1) throw new ArgumentException("array.GetLength(1) (" + array.GetLength(1) + ") != this.Length1 (" + this.Length1 + ")");
            if (array2.GetLength(0) != Length0) throw new ArgumentException("array2.GetLength(0) (" + array2.GetLength(0) + ") != this.Length0 (" + this.Length0 + ")");
            if (array2.GetLength(1) != Length1) throw new ArgumentException("array2.GetLength(1) (" + array2.GetLength(1) + ") != this.Length1 (" + this.Length1 + ")");
            for (int i = 0; i < Length0; i++)
            {
                for (int j = 0; j < Length1; j++)
                {
                    action(this[i, j], array[i, j], array2[i, j]);
                }
            }
        }

        public void ModifyAll<U>(U[,] array, Func<T, U, T> converter)
        {
            if (array.GetLength(0) != Length0) throw new ArgumentException("array.GetLength(0) (" + array.GetLength(0) + ") != this.Length0 (" + this.Length0 + ")");
            if (array.GetLength(1) != Length1) throw new ArgumentException("array.GetLength(1) (" + array.GetLength(1) + ") != this.Length1 (" + this.Length1 + ")");
            for (int i = 0; i < Length0; i++)
                for (int j = 0; j < Length1; j++)
                    this[i, j] = converter(this[i, j], array[i, j]);
        }

        public void ModifyAll<U, V>(U[,] array, V[,] array2, Func<T, U, V, T> converter)
        {
            if (array.GetLength(0) != Length0) throw new ArgumentException("array.GetLength(0) (" + array.GetLength(0) + ") != this.Length0 (" + this.Length0 + ")");
            if (array.GetLength(1) != Length1) throw new ArgumentException("array.GetLength(1) (" + array.GetLength(1) + ") != this.Length1 (" + this.Length1 + ")");
            if (array2.GetLength(0) != Length0) throw new ArgumentException("array2.GetLength(0) (" + array2.GetLength(0) + ") != this.Length0 (" + this.Length0 + ")");
            if (array2.GetLength(1) != Length1) throw new ArgumentException("array2.GetLength(1) (" + array2.GetLength(1) + ") != this.Length1 (" + this.Length1 + ")");
            for (int i = 0; i < Length0; i++)
                for (int j = 0; j < Length1; j++)
                    this[i, j] = converter(this[i, j], array[i, j], array2[i, j]);
        }
    }

#if false
    /// <summary>
    /// Special case when T is a reference type but treated like a value.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public class RefArray2D<T> : Array2D<T>
        where T : class, ICloneable, SettableTo<T>
    {
        public RefArray2D(int nRows, int nColumns)
            : base(nRows, nColumns)
        {
        }
        public RefArray2D(T[,] values)
            : base(values)
        {
        }
        public RefArray2D(RefArray2D<T> that)
            : base(that)
        {
        }

        protected void InitializeTo(T[,] array)
        {
            if (array.GetLength(0) != Length0) throw new ArgumentException("array.GetLength(0) (" + array.GetLength(0) + ") != this.Length0 (" + this.Length0 + ")");
            if (array.GetLength(1) != Length1) throw new ArgumentException("array.GetLength(1) (" + array.GetLength(1) + ") != this.Length1 (" + this.Length1 + ")");
            for (int i = 0; i < Length0; i++) {
                for (int j = 0; j < Length1; j++) {
                    this[i, j] = (T)array[i, j].Clone();
                }
            }
        }

        public override void SetTo(T[,] array)
        {
            if (array.GetLength(0) != Length0) throw new ArgumentException("array.GetLength(0) (" + array.GetLength(0) + ") != this.Length0 (" + this.Length0 + ")");
            if (array.GetLength(1) != Length1) throw new ArgumentException("array.GetLength(1) (" + array.GetLength(1) + ") != this.Length1 (" + this.Length1 + ")");
            if (array.Length == 0) return;
            if (this.array[0] == null) {
                InitializeTo(array);
            } else {
#if false
                ModifyAll(array, delegate(T item, T arrayItem) { item.SetTo(arrayItem); return item; });
#else
                for (int i = 0; i < Length0; i++) {
                    for (int j = 0; j < Length1; j++) {
                        this[i, j].SetTo(array[i, j]);
                    }
                }
#endif
            }
        }

        public override void CopyTo(T[,] array)
        {
            if (array.GetLength(0) != Length0) throw new ArgumentException("array.GetLength(0) (" + array.GetLength(0) + ") != this.Length0 (" + this.Length0 + ")");
            if (array.GetLength(1) != Length1) throw new ArgumentException("array.GetLength(1) (" + array.GetLength(1) + ") != this.Length1 (" + this.Length1 + ")");
            if (array.Length == 0) return;
            if (array[0, 0] == null) {
                for (int i = 0; i < Length0; i++) {
                    for (int j = 0; j < Length1; j++) {
                        array[i, j] = (T)this[i, j].Clone();
                    }
                }
            } else {
                SetItemsOf(array);
            }
        }

        protected void SetItemsOf(T[,] array)
        {
            if (array.GetLength(0) != Length0) throw new ArgumentException("array.GetLength(0) (" + array.GetLength(0) + ") != this.Length0 (" + this.Length0 + ")");
            if (array.GetLength(1) != Length1) throw new ArgumentException("array.GetLength(1) (" + array.GetLength(1) + ") != this.Length1 (" + this.Length1 + ")");
            for (int i = 0; i < Length0; i++) {
                for (int j = 0; j < Length1; j++) {
                    array[i, j].SetTo(this[i, j]);
                }
            }
        }

        public override void CopyTo(T[] array, int arrayIndex)
        {
            if (this.array.Length == 0) return;
            if (array[arrayIndex] == null) {
                for (int i = 0; i < this.array.Length; i++) {
                    array[i + arrayIndex] = (T)this.array[i].Clone();
                }
            } else {
                for (int i = 0; i < this.array.Length; i++) {
                    array[i + arrayIndex].SetTo(this.array[i]);
                }
            }
        }

        /// <summary>
        /// Clone the array and the items in the array.
        /// </summary>
        /// <returns></returns>
        public override object Clone()
        {
            return new RefArray2D<T>(this);
        }

    }
#endif
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}