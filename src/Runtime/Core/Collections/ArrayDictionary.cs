// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Collections
{
    using System.Collections;
    using System.Collections.Generic;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// Represents a dictionary with non-negative integers as keys.
    /// </summary>
    /// <remarks>
    /// This structure is efficient when the keys are zero-based and consecutive,
    /// since the dictionary is represented by an array under the hood.
    /// </remarks>
    /// <typeparam name="T">The type of a value.</typeparam>
    public class ArrayDictionary<T> : IEnumerable<KeyValuePair<int, T>>
    {
        /// <summary>
        /// An array of values, indexed by key.
        /// </summary>
        private readonly List<T> keyToValue;

        /// <summary>
        /// An array of flags indicating whether the corresponding key is present in the dictionary.
        /// </summary>
        private readonly List<bool> keyPresent;

        /// <summary>
        /// Initializes a new instance of the <see cref="ArrayDictionary{T}"/> class.
        /// </summary>
        public ArrayDictionary()
        {
            this.keyToValue = new List<T>();
            this.keyPresent = new List<bool>();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ArrayDictionary{T}"/> class with a given capacity.
        /// </summary>
        /// <param name="capacity">The capacity.</param>
        public ArrayDictionary(int capacity)
        {
            this.keyToValue = new List<T>(capacity);
            this.keyPresent = new List<bool>(capacity);
        }

        /// <summary>
        /// Gets or sets the value associated with a given key.
        /// </summary>
        /// <param name="key">The key of the value to get or set.</param>
        /// <returns>The value associated with the specified key.</returns>
        public T this[int key]
        {
            get
            {
                T value;
                if (!this.TryGetValue(key, out value))
                {
                    throw new KeyNotFoundException();
                }

                return value;
            }

            set { this.DoAdd(key, value, false); }
        }

        /// <summary>
        /// Gets the value associated with a given key.
        /// </summary>
        /// <param name="key">The key of the value to get.</param>
        /// <param name="value">
        /// When this method returns, contains the value associated with the specified key, if the key is found;
        /// otherwise, the default value for the type of the <paramref name="value"/> parameter.
        /// </param>
        /// <returns>
        /// <see langword="true"/> if the dictionary contains an element with the specified key;
        /// otherwise, <see langword="false"/>.
        /// </returns>
        public bool TryGetValue(int key, out T value)
        {
            Argument.CheckIfInRange(key >= 0, "key", "A key must be non-negative.");

            if (!this.ContainsKey(key))
            {
                value = default(T);
                return false;
            }

            value = this.keyToValue[key];
            return true;
        }

        /// <summary>
        /// Adds a given key and value to the dictionary.
        /// </summary>
        /// <param name="key">The key of the element to add.</param>
        /// <param name="value">The value of the element to add. The value can be <see langword="null"/> for reference types.</param>
        public void Add(int key, T value)
        {
            this.DoAdd(key, value, true);
        }

        /// <summary>
        /// Determines whether the dictionary contains a given key.
        /// </summary>
        /// <param name="key">The key to locate.</param>
        /// <returns>
        /// <see langword="true"/> if the dictionary contains an element with the specified key;
        /// otherwise, <see langword="false"/>.
        /// </returns>
        public bool ContainsKey(int key)
        {
            return key >= 0 && key < this.keyToValue.Count && this.keyPresent[key];
        }

        /// <summary>
        /// Returns an enumerator that iterates through the dictionary in ascending key order.
        /// </summary>
        /// <returns>An enumerator for the dictionary.</returns>
        public IEnumerator<KeyValuePair<int, T>> GetEnumerator()
        {
            for (int i = 0; i < this.keyToValue.Count; ++i)
            {
                if (this.keyPresent[i])
                {
                    yield return new KeyValuePair<int, T>(i, this.keyToValue[i]);
                }
            }
        }

        /// <summary>
        /// Returns an enumerator that iterates through the dictionary in ascending key order.
        /// </summary>
        /// <returns>An enumerator for the dictionary.</returns>
        IEnumerator IEnumerable.GetEnumerator()
        {
            return this.GetEnumerator();
        }

        /// <summary>
        /// Adds a given key and value to the dictionary.
        /// </summary>
        /// <param name="key">The key of the element to add.</param>
        /// <param name="value">The value of the element to add. The value can be <see langword="null"/> for reference types.</param>
        /// <param name="throwIfKeyPresent">Specifies whether the method should throw if the key is already present in the dictionary.</param>
        private void DoAdd(int key, T value, bool throwIfKeyPresent)
        {
            Argument.CheckIfInRange(key >= 0, "key", "A key must be non-negative.");
            Argument.CheckIfValid(!throwIfKeyPresent || !this.ContainsKey(key), "key", "The given key is already presented in the dictionary.");

            for (int i = this.keyToValue.Count; i <= key; ++i)
            {
                this.keyToValue.Add(default(T));
                this.keyPresent.Add(false);
            }

            this.keyToValue[key] = value;
            this.keyPresent[key] = true;
        }
    }
}