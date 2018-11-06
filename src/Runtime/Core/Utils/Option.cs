// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Utilities
{
    using System;

    using Microsoft.ML.Probabilistic.Serialization;

    /// <summary>
    /// Represents a value type that can be absent
    /// </summary>
    /// <remarks>
    /// Unlike <see cref="Nullable{T}"/> this works with value and reference types.
    /// </remarks>
    public struct Option<T>
    {
        private readonly T value;

        /// <summary>
        /// Initializes a new instance of the <see cref="Option{T}"/> structure.
        /// If T is reference type and <see cref="value"/> is null then empty option is crated.
        /// </summary>
        [Construction("Value")]
        public Option(T value)
        {
            this.value = value;
            this.HasValue = value != null;
        }

        /// <summary>
        /// Gets a value indicating whether the current <see cref="Option{T}" /> object
        /// has a valid value of its underlying type.
        /// </summary>
        public bool HasValue { get; }

        /// <summary>
        /// Gets a value indicating whether the current <see cref="Option{T}" /> object
        /// has no valid value of its underlying type.
        /// </summary>
        /// <remarks>
        /// This property is redundant but is needed for quoting operation.
        /// </remarks>
        public bool HasNoValue => !this.HasValue;

        /// <summary>
        /// Gets the value of the current <see cref="Option{T}" /> object
        /// if it has been assigned a valid underlying value.
        /// </summary>
        /// <returns>
        /// The value of the current <see cref="Option{T}" /> object if the <see cref="HasValue"/>
        /// property is true. An exception is thrown otherwise.
        /// </returns>
        public T Value =>
            this.HasValue
                ? this.value
                : throw new InvalidOperationException($"Can't get value of empty {nameof(Option<T>)}");

        /// <summary>
        /// Creates a news <see cref="Option{T}"/> object which holds no value.
        /// </summary>
        [Construction(UseWhen = "HasNoValue")]
        public static Option<T> Empty() => new Option<T>();

        /// <summary>
        /// Creates a new <see cref="Option{T}" /> object initialized to a specified value.
        /// </summary>
        public static implicit operator Option<T>(T value) => new Option<T>(value);

        /// <summary>
        /// Creates a new empty <see cref="Option{T}" /> object.
        /// </summary>
        public static implicit operator Option<T>(Option.NoneType none) => new Option<T>();

        /// <summary>
        /// Defines an explicit conversion of a <see cref="Option{T}" /> instance to its underlying value.
        /// </summary>
        public static explicit operator T(Option<T> value) => value.value;

        public override bool Equals(object other) =>
            this.HasValue
                ? other != null && this.value.Equals(other)
                : other == null;

        public override int GetHashCode() => this.HasValue ? this.value.GetHashCode() : 0;

        public override string ToString() => this.HasValue ? this.value.ToString() : string.Empty;
    }

    /// <summary>
    /// Helper static class with constructor methods for <see cref="Option{T}"/>.
    /// </summary>
    public static class Option
    {
        /// <summary>
        /// Creates a new empty <see cref="Option{T}" /> object.
        /// </summary>
        /// <remarks>
        /// Since at the call site generic type argument T of the <see cref="Option{T}"/> is not known
        /// a special sentinel value of <see cref="NoneType"/> is returned. It is implicitly casted
        /// into any <see cref="Option{T}"/> on demand.
        /// </remarks>
        public static NoneType None => default(NoneType);

        /// <summary>
        /// Creates a new <see cref="Option{T}" /> object initialized to a specified value.
        /// </summary>
        /// <remarks>
        /// Throws <see cref="ArgumentNullException"/> if <paramref name="value"/> is null.
        /// <see cref="Some{T}"/> constructor should be used only in places where value
        /// is expected to be non-null.
        /// </remarks>
        public static Option<T> Some<T>(T value)
        {
            if (value == null)
            {
                throw new ArgumentNullException(
                    nameof(value),
                    $"Option.Some constructor can be used only with non-null values");
            }

            return new Option<T>(value);
        }

        /// <summary>
        /// Helper class to represent "any absent value" which can be converted
        /// into any <see cref="Option{T}"/>.
        /// </summary>
        public struct NoneType
        {
        }
    }
}
