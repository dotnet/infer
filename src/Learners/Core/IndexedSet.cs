// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Collections
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using Microsoft.ML.Probabilistic.Serialization;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// A bidirectional mapping of unique elements to non-negative indexes.
    /// </summary>
    /// <typeparam name="T">The type of an element.</typeparam>
    [Serializable]
    public class IndexedSet<T>
    {
        #region Fields, constructors, properties

        /// <summary>
        /// The current custom binary serialization version of the <see cref="IndexedSet{T}"/> class.
        /// </summary>
        private const int CustomSerializationVersion = 1;

        /// <summary>
        /// A mapping from an element to its index.
        /// </summary>
        private Dictionary<T, int> elementToIndex;

        /// <summary>
        /// A mapping from an index to its element.
        /// </summary>
        private List<T> indexToElement;

        /// <summary>
        /// Initializes a new instance of the <see cref="IndexedSet{T}"/> class.
        /// </summary>
        public IndexedSet()
        {
            this.elementToIndex = new Dictionary<T, int>();
            this.indexToElement = new List<T>();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="IndexedSet{T}"/> class.
        /// </summary>
        /// <param name="elements">A collection of elements.</param>
        public IndexedSet(IEnumerable<T> elements) : this()
        {
            if (elements == null)
            {
                throw new ArgumentNullException(nameof(elements));
            }

            foreach (var element in elements)
            {
                this.Add(element);
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="IndexedSet{T}"/> class
        /// from a reader of a binary stream.
        /// </summary>
        /// <param name="reader">The reader to load the indexed set from.</param>
        public IndexedSet(IReader reader) : this()
        {
            if (reader == null)
            {
                throw new ArgumentNullException(nameof(reader));
            }

            int deserializedVersion = reader.ReadSerializationVersion(CustomSerializationVersion);

            if (deserializedVersion == CustomSerializationVersion)
            {
                int elementCount = reader.ReadInt32();
                for (int index = 0; index < elementCount; index++)
                {
                    var element = reader.ReadObject<T>();
                    this.Add(element, false);
                }
            }
        }

        /// <summary>
        /// Gets the number of elements in the indexed set.
        /// </summary>
        public int Count
        {
            get
            {
                Debug.Assert(
                    this.indexToElement.Count == this.elementToIndex.Count,
                    "Element-to-index and index-to-element mappings must contain the same number of elements.");

                return this.indexToElement.Count;
            }
        }

        /// <summary>
        /// Gets all elements in the indexed set.
        /// </summary>
        public IEnumerable<T> Elements
        {
            get
            {
                return this.indexToElement;
            }
        }

        /// <summary>
        /// Gets the indexes of all elements in the indexed set.
        /// </summary>
        public IEnumerable<int> Indexes
        {
            get
            {
                return this.elementToIndex.Values;
            }
        }

        #endregion

        #region Methods

        /// <summary>
        /// Adds a new element to the indexed set.
        /// </summary>
        /// <param name="element">The element to add.</param>
        /// <param name="throwIfPresent">
        /// If true, an exception is throw when adding an element that is already present in the indexed set. Defaults to true.
        /// </param>
        /// <returns>The index of the added element.</returns>
        public int Add(T element, bool throwIfPresent = true)
        {
            if (element == null)
            {
                throw new ArgumentNullException(nameof(element));
            }

            int index;
            if (this.elementToIndex.TryGetValue(element, out index))
            {
                if (throwIfPresent)
                {
                    throw new ArgumentException("All elements must be unique.", nameof(element));
                }

                return index;
            }

            index = this.Count;
            this.elementToIndex.Add(element, index);
            this.indexToElement.Add(element);

            return index;
        }

        /// <summary>
        /// Gets an element for a specified index.
        /// </summary>
        /// <param name="index">The index to get the element for.</param>
        /// <returns>The element.</returns>
        public T GetElementByIndex(int index)
        {
            if (index < 0 && index >= this.Count)
            {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            return this.indexToElement[index];
        }

        /// <summary>
        /// Gets the index for a specified element.
        /// </summary>
        /// <param name="element">The element to get the index for.</param>
        /// <param name="index">The index for the specified element, if present in the set.</param>
        /// <returns>True if <paramref name="element"/> is present in the indexed set and false otherwise.</returns>
        public bool TryGetIndex(T element, out int index)
        {
            if (element == null)
            {
                throw new ArgumentNullException(nameof(index));
            }

            return this.elementToIndex.TryGetValue(element, out index);
        }

        /// <summary>
        /// Returns true if the specified element is contained in the indexed set and false otherwise.
        /// </summary>
        /// <param name="element">The element.</param>
        /// <returns>True if the specified element is contained in the indexed set and false otherwise.</returns>
        public bool Contains(T element)
        {
            if (element == null)
            {
                throw new ArgumentNullException(nameof(element));
            }

            return this.elementToIndex.ContainsKey(element);
        }

        /// <summary>
        /// Removes all elements from the indexed set.
        /// </summary>
        public void Clear()
        {
            this.elementToIndex = new Dictionary<T, int>();
            this.indexToElement = new List<T>();
        }

        /// <summary>
        /// Saves the elements of the indexed set to a binary writer.
        /// </summary>
        /// <param name="writer">The writer to save the elements of the indexed set to.</param>
        public void SaveForwardCompatible(IWriter writer)
        {
            if (writer == null)
            {
                throw new ArgumentNullException(nameof(writer));
            }

            writer.Write(CustomSerializationVersion);
            writer.Write(this.Count);
            for (int index = 0; index < this.Count; index++)
            {
                writer.WriteObject(this.indexToElement[index]);
            }
        }

        #endregion
    }
}
