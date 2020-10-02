// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Collections
{
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;
    using System.Runtime.Serialization;

    using Microsoft.ML.Probabilistic.Serialization;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// Represents an immutable array.
    /// </summary>
    /// <remarks>
    /// This is a partial reimplementation of System.Collections.Immutable.ReadOnlyArray.
    /// Once we can move to netcore-only codebase, this type can be removed.
    /// API is supposed to be a subset of the real thing to ease migration in future.
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
        private ReadOnlyArray(T[] array)
        {
            this.array = array;
        }

        /// <summary>
        /// Creates a new instance of <see cref="ReadOnlyArray{T}"/> by copying elements of
        /// <paramref name="sequence"/>.
        /// </summary>
        [Construction("CloneArray")]
        public static ReadOnlyArray<T> CreateCopy(IEnumerable<T> sequence) =>
            new ReadOnlyArray<T>(sequence.ToArray());

        public static ReadOnlyArray<T> Empty => new ReadOnlyArray<T>(Array.Empty<T>());

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

        public override bool Equals(object o) => o is ReadOnlyArray<T> that && this == that;

        public override int GetHashCode() => this.array.GetHashCode();

        public static bool operator ==(ReadOnlyArray<T> left, ReadOnlyArray<T> right) =>
            left.array == right.array;

        public static bool operator !=(ReadOnlyArray<T> left, ReadOnlyArray<T> right) =>
            left.array != right.array;

        public class Builder
        {
            private T[] array;

            public Builder(int size) =>
                this.array = new T[size];

            public T this[int index]
            {
                get => this.array[index];
                set => this.array[index] = value;
            }

            public int Count => this.array.Length;

            public ReadOnlyArray<T> MoveToImmutable()
            {
                var result = new ReadOnlyArray<T>(this.array);
                this.array = Array.Empty<T>();
                return result;
            }
        }
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
    /// Enumerator for immutable arrays and immutable array segments.
    /// </summary>
    public struct ReadOnlyArraySegmentEnumerator<T> : IEnumerator<T>
    {
        /// <summary>
        /// Underlying immutable array.
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

    public static class ReadOnlyArray
    {
        public static ReadOnlyArray<T>.Builder CreateBuilder<T>(int size) =>
            new ReadOnlyArray<T>.Builder(size);

        public static ReadOnlyArray<T> Create<T>() => ReadOnlyArray<T>.Empty;

        public static ReadOnlyArray<T> Create<T>(T elem)
        {
            var builder = new ReadOnlyArray<T>.Builder(1) {[0] = elem};
            return builder.MoveToImmutable();
        }

        public static ReadOnlyArray<T> Create<T>(T elem1, T elem2)
        {
            var builder = new ReadOnlyArray<T>.Builder(2) {[0] = elem1, [1] = elem2};
            return builder.MoveToImmutable();
        }

        /// <summary>
        /// Syntactic sugar for `ReadOnlyArray{T}.CreateCopy(sequence)`
        /// </summary>
        public static ReadOnlyArray<T> ToReadOnlyArray<T>(this IEnumerable<T> sequence) =>
            ReadOnlyArray<T>.CreateCopy(sequence);
    }
}
