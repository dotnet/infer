// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Collections
{
    using Microsoft.ML.Probabilistic.Serialization;
    using System;
    using System.Runtime.Serialization;

    using Utilities;

    /// <summary>
    /// Stores a value and its index in the collection.
    /// </summary>
    /// <typeparam name="T">The type of the stored value.</typeparam>
    [Serializable]
    [DataContract]
    public struct ValueAtIndex<T>
    {
        /// <summary>
        /// Represents value missing from a collection.
        /// </summary>
        public static ValueAtIndex<T> NoElement = new ValueAtIndex<T> {Index = -1};

        #region Constructors

        /// <summary>
        /// Initializes a new instance of the <see cref="ValueAtIndex{T}"/> struct.
        /// </summary>
        /// <param name="index">The value of the collection element.</param>
        /// <param name="value">The index of the collection element.</param>
        [Construction("Index", "Value")]
        public ValueAtIndex(int index, T value)
            : this()
        {
            this.Index = index;
            this.Value = value;
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets or sets the index of the element.
        /// </summary>
        [DataMember]
        public int Index { get; set; }

        /// <summary>
        /// Gets or sets the value of the element.
        /// </summary>
        [DataMember]
        public T Value { get; set; }

        #endregion

        #region ToString

        /// <summary>
        /// Converts the value of this instance to its equivalent string representation.
        /// </summary>
        /// <returns>The string representation of the value of this instance.</returns>
        public override string ToString()
        {
            return "[" + this.Index + "]=" + this.Value;
        }

        #endregion
    }
}