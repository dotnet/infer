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
    /// This is a partial reimplementation of System.Collections.Immutable.ImmutableArray.
    /// Once we can move to netcore-only codebase, this type can be removed.
    /// API is supposed to be a subset of the real thing to ease migration in future.
    /// </remarks>
    [Serializable]
    [DataContract]
    public struct ImmutableArray<T> : IReadOnlyList<T>
    {
        /// <summary>
        /// Regular array that holds data.
        /// </summary>
        [DataMember]
        private readonly T[] array;

        /// <summary>
        /// Initializes a new instance of the <see cref="ImmutableArray{T}"/> structure.
        /// </summary>
        private ImmutableArray(T[] array)
        {
            this.array = array;
        }

        /// <summary>
        /// Creates a new instance of <see cref="ImmutableArray{T}"/> by copying elements of
        /// <paramref name="sequence"/>.
        /// </summary>
        [Construction("CloneArray")]
        public static ImmutableArray<T> CreateCopy(IEnumerable<T> sequence) =>
            new ImmutableArray<T>(sequence.ToArray());

        public static ImmutableArray<T> Empty => new ImmutableArray<T>(Array.Empty<T>());

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
        public ImmutableArraySegmentEnumerator<T> GetEnumerator() =>
            new ImmutableArraySegmentEnumerator<T>(this, 0, this.array.Length);

        /// <inheritdoc/>
        IEnumerator<T> IEnumerable<T>.GetEnumerator() =>
            new ImmutableArraySegmentEnumerator<T>(this, 0, this.array.Length);

        /// <inheritdoc/>
        IEnumerator IEnumerable.GetEnumerator() =>
            new ImmutableArraySegmentEnumerator<T>(this, 0, this.array.Length);

        public override bool Equals(object o) => o is ImmutableArray<T> that && this == that;

        public override int GetHashCode() => this.array.GetHashCode();

        public static bool operator ==(ImmutableArray<T> left, ImmutableArray<T> right) =>
            left.array == right.array;

        public static bool operator !=(ImmutableArray<T> left, ImmutableArray<T> right) =>
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

            public ImmutableArray<T> MoveToImmutable()
            {
                var result = new ImmutableArray<T>(this.array);
                this.array = Array.Empty<T>();
                return result;
            }
        }
    }

    /// <summary>
    /// A version if <see cref="ArraySegment{T}"/> which can not be mutated.
    /// </summary>
    public struct ImmutableArraySegment<T> : IReadOnlyList<T>
    {
        /// <summary>
        /// Underlying read-only array.
        /// </summary>
        private readonly ImmutableArray<T> array;

        /// <summary>
        /// Index of the first element which belongs to this segment.
        /// </summary>
        private readonly int begin;

        /// <summary>
        /// Index of the first element which does not belong to this segment.
        /// </summary>
        private readonly int length;

        /// <summary>
        /// Initializes a new instance of <see cref="ImmutableArraySegment{T}"/> structure.
        /// </summary>
        public ImmutableArraySegment(ImmutableArray<T> array, int begin, int length)
        {
            Argument.CheckIfInRange(begin >= 0 && begin <= array.Count, nameof(begin), "Segment begin should be in the range [0, array.Count]");
            Argument.CheckIfInRange(length >= 0 && length <= array.Count - begin, nameof(length), "Segment length should be in the range [0, array.Count - begin]");

            this.array = array;
            this.begin = begin;
            this.length = length;
        }

        [Construction("CloneArray")]
        public static ImmutableArraySegment<T> CreateCopy(IEnumerable<T> sequence)
        {
            var array = ImmutableArray<T>.CreateCopy(sequence);
            return new ImmutableArraySegment<T>(array, 0, array.Count);
        }

        public static implicit operator ImmutableArraySegment<T>(ImmutableArray<T> array) =>
            new ImmutableArraySegment<T>(array, 0, array.Count);

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

        public ImmutableArray<T> BaseArray => this.array;

        public int BaseIndex => this.begin;

        public T[] CloneArray()
        {
            var result = new T[this.Count];
            for (var i = 0; i < this.Count; ++i)
            {
                result[i] = this[i];
            }

            return result;
        }

        /// <summary>
        /// Returns enumerator over elements of array.
        /// </summary>
        /// <remarks>
        /// This is value-type non-virtual version of enumerator that is used by compiler in foreach loops.
        /// </remarks>
        public ImmutableArraySegmentEnumerator<T> GetEnumerator() =>
            new ImmutableArraySegmentEnumerator<T>(this.array, this.begin, this.begin + this.length);

        /// <inheritdoc/>
        IEnumerator<T> IEnumerable<T>.GetEnumerator() =>
            new ImmutableArraySegmentEnumerator<T>(this.array, this.begin, this.begin + this.length);

        /// <inheritdoc/>
        IEnumerator IEnumerable.GetEnumerator() =>
            new ImmutableArraySegmentEnumerator<T>(this.array, this.begin, this.begin + this.length);
    }

    /// <summary>
    /// Enumerator for immutable arrays and immutable array segments.
    /// </summary>
    public struct ImmutableArraySegmentEnumerator<T> : IEnumerator<T>
    {
        /// <summary>
        /// Underlying immutable array.
        /// </summary>
        private readonly ImmutableArray<T> array;

        /// <summary>
        /// Index of the first element which does not belong segment begin enumerated.
        /// </summary>
        private readonly int end;

        /// <summary>
        /// Index of the current element
        /// </summary>
        private int pointer;

        /// <summary>
        /// Initializes a new instance of <see cref="ImmutableArraySegment{T}"/> structure.
        /// </summary>
        internal ImmutableArraySegmentEnumerator(ImmutableArray<T> array, int begin, int end)
        {
            this.array = array;
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
        void IEnumerator.Reset() => this.Reset();

        public void Reset() => throw new NotSupportedException();
    }

    public static class ImmutableArray
    {
        public static ImmutableArray<T>.Builder CreateBuilder<T>(int size) =>
            new ImmutableArray<T>.Builder(size);

        public static ImmutableArray<T> Create<T>() => ImmutableArray<T>.Empty;

        public static ImmutableArray<T> Create<T>(T elem)
        {
            var builder = new ImmutableArray<T>.Builder(1) {[0] = elem};
            return builder.MoveToImmutable();
        }

        public static ImmutableArray<T> Create<T>(T elem1, T elem2)
        {
            var builder = new ImmutableArray<T>.Builder(2) {[0] = elem1, [1] = elem2};
            return builder.MoveToImmutable();
        }

        /// <summary>
        /// Syntactic sugar for `ReadOnlyArray{T}.CreateCopy(sequence)`
        /// </summary>
        public static ImmutableArray<T> ToImmutableArray<T>(this IEnumerable<T> sequence) =>
            ImmutableArray<T>.CreateCopy(sequence);
    }
}
