// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Compiler.Graphs
{
    /// <summary>
    /// Represents a collection of integers in sequence from LowerBound to UpperBound.
    /// </summary>
    internal class Range : IList<int>
    {
        protected int Start, Count;

        public int LowerBound
        {
            get { return Start; }
            set { Start = value; }
        }

        public int UpperBound
        {
            get { return Start + Count - 1; }
            set { Count = value - Start + 1; }
        }

        public Range(int start, int count)
        {
            this.Start = start;
            this.Count = count;
        }

        public void Add(int item)
        {
            throw new Exception("The method or operation is not implemented.");
        }

        public void Clear()
        {
            throw new Exception("The method or operation is not implemented.");
        }

        public bool Contains(int item)
        {
            return (item >= LowerBound) && (item <= UpperBound);
        }

        public void CopyTo(int[] array, int arrayIndex)
        {
            for (int i = 0; i < Count; i++)
            {
                array[arrayIndex + i] = Start + i;
            }
        }

        int ICollection<int>.Count
        {
            get { return Count; }
        }

        public bool IsReadOnly
        {
            get { return true; }
        }

        public bool Remove(int item)
        {
            throw new Exception("The method or operation is not implemented.");
        }

        public IEnumerator<int> GetEnumerator()
        {
            for (int i = 0; i < Count; i++)
            {
                yield return Start + i;
            }
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public int IndexOf(int item)
        {
            if (Contains(item)) return item - LowerBound;
            else return -1;
        }

        public void Insert(int index, int item)
        {
            throw new Exception("The method or operation is not implemented.");
        }

        public void RemoveAt(int index)
        {
            throw new Exception("The method or operation is not implemented.");
        }

        public int this[int index]
        {
            get { return Start + index; }
            set { throw new Exception("The method or operation is not implemented."); }
        }
    }

    /// <summary>
    /// Represents a multidimensional grid of integer points.
    /// </summary>
    internal class MultiRange : IList<int[]>
    {
        public int[] LowerBounds, Lengths, Strides;

        public MultiRange(params int[] lengths)
            : this(new int[lengths.Length], lengths)
        {
        }

        public MultiRange(int[] lowerBounds, int[] lengths)
        {
            LowerBounds = lowerBounds;
            Lengths = lengths;
            Strides = StringUtil.ArrayStrides(lengths);
        }

        public static MultiRange ArrayIndices(Array array)
        {
            return new MultiRange(StringUtil.ArrayLowerBounds(array), StringUtil.ArrayDimensions(array));
        }

        public int Rank
        {
            get { return Lengths.Length; }
        }

        public int GetUpperBound(int dim)
        {
            return LowerBounds[dim] + Lengths[dim] - 1;
        }

        public int IndexOf(int[] item)
        {
            if (!Contains(item)) return -1;
            int index = 0;
            for (int i = 0; i < Rank; i++)
            {
                index += (item[i] - LowerBounds[i])*Strides[i];
            }
            return index;
        }

        public void Insert(int index, int[] item)
        {
            throw new Exception("The method or operation is not implemented.");
        }

        public void RemoveAt(int index)
        {
            throw new Exception("The method or operation is not implemented.");
        }

        public int[] this[int index]
        {
            get
            {
                int[] value = new int[Strides.Length];
                StringUtil.LinearIndexToMultidimensionalIndex(index, Strides, value, LowerBounds);
                return value;
            }
            set { throw new Exception("The method or operation is not implemented."); }
        }

        public void Add(int[] item)
        {
            throw new Exception("The method or operation is not implemented.");
        }

        public void Clear()
        {
            throw new Exception("The method or operation is not implemented.");
        }

        public bool Contains(int[] item)
        {
            for (int i = 0; i < Rank; i++)
            {
                if (!(item[i] >= LowerBounds[i] && item[i] < LowerBounds[i] + Lengths[i])) return false;
            }
            return true;
        }

        public void CopyTo(int[][] array, int arrayIndex)
        {
            throw new Exception("The method or operation is not implemented.");
        }

        public int Count
        {
            get
            {
                int count = 1;
                for (int i = 0; i < Rank; i++)
                {
                    count *= Lengths[i];
                }
                return count;
            }
        }

        public bool IsReadOnly
        {
            get { return true; }
        }

        public bool Remove(int[] item)
        {
            throw new Exception("The method or operation is not implemented.");
        }

        public IEnumerator<int[]> GetEnumerator()
        {
            int count = Count;
            int[] index = new int[Rank];
            for (int i = 0; i < count; i++)
            {
                StringUtil.LinearIndexToMultidimensionalIndex(i, Strides, index, LowerBounds);
                yield return index;
            }
        }

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}