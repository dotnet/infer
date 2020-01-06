// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Collections
{
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Runtime.Serialization;

    using Microsoft.ML.Probabilistic.Serialization;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// Represents a read only array.
    /// </summary>
    /// <remarks>
    /// It is implemented as struct because it avoids extra allocations on heap.
    /// <see cref="ReadOnlyArray{T}"/> doesn't have space overhead compared to regular arrays.
    /// </remarks>
    [Serializable]
    [DataContract]
    public struct ReadOnlyArray<T> : IReadOnlyList<T>
    {
        /// <summary>
        /// Regular array that holds data.
        /// </summary>
        [DataMember]
        private readonly T[] array;

        /// <summary>
        /// Initializes a new instance of the <see cref="ReadOnlyArray{T}"/> structure.
        /// </summary>
        [Construction("CloneArray")]
        public ReadOnlyArray(T[] array)
        {
            this.array = array;
        }

        /// <summary>
        /// Gets a boolean value which is true if this ReadOnlyArray wraps null array.
        /// </summary>
        public bool IsNull => this.array == null;

        /// <inheritdoc/>
        public T this[int index] => this.array[index];

        /// <inheritdoc/>
        public int Count => this.array.Length;

        /// <summary>
        /// Creates a mutable copy of this array.
        /// </summary>
        public T[] CloneArray() => (T[])this.array.Clone();

        /// <summary>
        /// Returns enumerator over elements of array.
        /// </summary>
        /// <remarks>
        /// This is value-type non-virtual version of enumerator that is used by compiler in foreach loops.
        /// </remarks>
        public ReadOnlyArraySegmentEnumerator<T> GetEnumerator() =>
            new ReadOnlyArraySegmentEnumerator<T>(this, 0, this.array.Length);

        /// <inheritdoc/>
        IEnumerator<T> IEnumerable<T>.GetEnumerator() =>
            new ReadOnlyArraySegmentEnumerator<T>(this, 0, this.array.Length);

        /// <inheritdoc/>
        IEnumerator IEnumerable.GetEnumerator() =>
            new ReadOnlyArraySegmentEnumerator<T>(this, 0, this.array.Length);

        /// <summary>
        /// Helper method which allows to cast regular arrays to read only versions implicitly.
        /// </summary>
        public static implicit operator ReadOnlyArray<T>(T[] array) => new ReadOnlyArray<T>(array);
    }

    /// <summary>
    /// A version if <see cref="ArraySegment{T}"/> which can not be mutated.
    /// </summary>
    public struct ReadOnlyArraySegment<T> : IReadOnlyList<T>
    {
        /// <summary>
        /// Underlying read-only array.
        /// </summary>
        private readonly ReadOnlyArray<T> array;

        /// <summary>
        /// Index of the first element which belongs to this segment.
        /// </summary>
        private readonly int begin;

        /// <summary>
        /// Index of the first element which does not belong to this segment.
        /// </summary>
        private readonly int length;

        /// <summary>
        /// Initializes a new instance of <see cref="ReadOnlyArraySegment{T}"/> structure.
        /// </summary>
        public ReadOnlyArraySegment(ReadOnlyArray<T> array, int begin, int length)
        {
            Argument.CheckIfValid(!array.IsNull, nameof(array));
            Argument.CheckIfInRange(begin >= 0 && begin <= array.Count, nameof(begin), "Segment begin should be in the range [0, array.Count]");
            Argument.CheckIfInRange(length >= 0 && length <= array.Count - begin, nameof(length), "Segment length should be in the range [0, array.Count - begin]");

            this.array = array;
            this.begin = begin;
            this.length = length;
        }

        /// <inheritdoc/>
        public T this[int index]
        {
            get
            {
                Debug.Assert(index >= 0 && index < this.length);
                return this.array[this.begin + index];
            }
        }

        /// <inheritdoc/>
        public int Count => this.length;

        /// <summary>
        /// Returns enumerator over elements of array.
        /// </summary>
        /// <remarks>
        /// This is value-type non-virtual version of enumerator that is used by compiler in foreach loops.
        /// </remarks>
        public ReadOnlyArraySegmentEnumerator<T> GetEnumerator() =>
            new ReadOnlyArraySegmentEnumerator<T>(this.array, this.begin, this.begin + this.length);

        /// <inheritdoc/>
        IEnumerator<T> IEnumerable<T>.GetEnumerator() =>
            new ReadOnlyArraySegmentEnumerator<T>(this.array, this.begin, this.begin + this.length);

        /// <inheritdoc/>
        IEnumerator IEnumerable.GetEnumerator() =>
            new ReadOnlyArraySegmentEnumerator<T>(this.array, this.begin, this.begin + this.length);
    }

    /// <summary>
    /// Enumerator for read only arrays and read only array segments.
    /// </summary>
    public struct ReadOnlyArraySegmentEnumerator<T> : IEnumerator<T>
    {
        /// <summary>
        /// Underlying read-only array.
        /// </summary>
        private readonly ReadOnlyArray<T> array;

        /// <summary>
        /// Index of the first element which belongs segment begin enumerated.
        /// </summary>
        private readonly int begin;

        /// <summary>
        /// Index of the first element which does not belong segment begin enumerated.
        /// </summary>
        private readonly int end;

        /// <summary>
        /// Index of the current element
        /// </summary>
        private int pointer;

        /// <summary>
        /// Initializes a new instance of <see cref="ReadOnlyArraySegment{T}"/> structure.
        /// </summary>
        internal ReadOnlyArraySegmentEnumerator(ReadOnlyArray<T> array, int begin, int end)
        {
            this.array = array;
            this.begin = begin;
            this.end = end;
            this.pointer = begin - 1;
        }

        /// <inheritdoc/>
        public void Dispose()
        {
        }

        /// <inheritdoc/>
        public bool MoveNext()
        {
            ++this.pointer;
            return this.pointer < this.end;
        }

        /// <inheritdoc/>
        public T Current => this.array[this.pointer];

        /// <inheritdoc/>
        object IEnumerator.Current => this.Current;

        /// <inheritdoc/>
        void IEnumerator.Reset()
        {
            this.pointer = this.begin - 1;
        }
    }
}
