// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Collections
{
    /// <summary>
    /// Hash table implementation that has a constant-time Clear() operation.
    /// </summary>
    public class GenerationalDictionary<TKey, TValue>
        where TKey : IEquatable<TKey>
        where TValue : unmanaged
    {
        private const int OccupiedMask = 1 << 31;

        struct Entry
        {
            public TKey Key;
            public int Hash;
            public int Generation;
            public TValue Value;
        };

        private Entry[] entries;
        private int generation;
        private int growThreshold;
        private int mask;
        private int filledEntriesCount;

        public GenerationalDictionary()
        {
            this.Initialize();
        }

        public int Count => this.filledEntriesCount;

        public TValue this[TKey key] =>
            this.TryGetValue(key, out var value) ? value : throw new KeyNotFoundException("Value not found");

        public bool TryGetValue(TKey key, out TValue value)
        {
            var currentGeneration = this.generation;
            var hash = key.GetHashCode() | OccupiedMask;
            var index = hash & this.mask;
            for (var probe = 1;; ++probe)
            {
                if (this.entries[index].Generation != currentGeneration)
                {
                    value = default(TValue);
                    return false;
                }

                if (this.entries[index].Hash == hash &&
                    this.entries[index].Key.Equals(key))
                {
                    value = this.entries[index].Value;
                    return true;
                }

                index = (index + probe) & this.mask;
            }
        }

        public void Add(TKey key, TValue value)
        {
            if (this.filledEntriesCount >= this.growThreshold)
            {
                this.Grow();
            }

            var currentGeneration = this.generation;
            var hash = key.GetHashCode() | OccupiedMask;
            var index = hash & this.mask;
            for (var probe = 1;; ++probe)
            {
                if (this.entries[index].Generation != currentGeneration)
                {
                    ++this.filledEntriesCount;

                    this.entries[index] = new Entry
                    {
                        Key = key,
                        Hash = hash,
                        Generation = this.generation,
                        Value = value,
                    };

                    return;
                }

                if (this.entries[index].Hash == hash &&
                    this.entries[index].Key.Equals(key))
                {
                    throw new ArgumentException("Element with given key already exists in the dictionary");
                }

                index = (index + probe) & this.mask;
            } 
        }

        public void Update(TKey key, TValue value)
        {
            var currentGeneration = this.generation;
            var hash = key.GetHashCode() | OccupiedMask;
            var index = hash & this.mask;
            for (var probe = 1;; ++probe)
            {
                if (this.entries[index].Generation != currentGeneration)
                {
                    throw new ArgumentException("Element with given key already does not exist in the dictionary");
                }

                if (this.entries[index].Hash == hash &&
                    this.entries[index].Key.Equals(key))
                {
                    this.entries[index].Value = value;
                    return;
                }

                index = (index + probe) & this.mask;
            } 
        }

        public void Clear()
        {
            this.filledEntriesCount = 0;
            ++this.generation;
            if (this.generation == 0)
            {
                // If generation wraps-around (highly unlikely), recreate dictionary from scratch
                this.Initialize();
            }
        }

        private void Grow()
        {
            var oldEntries = this.entries;
            var currentGeneration = this.generation;
            this.entries = new Entry[oldEntries.Length * 2];
            this.growThreshold = this.entries.Length / 2;
            this.mask = this.entries.Length - 1;

            foreach (var entry in oldEntries)
            {
                if (entry.Generation != currentGeneration)
                {
                    continue;
                }

                var index = entry.Hash & this.mask;
                for (var probe = 1; this.entries[index].Hash != 0; ++probe)
                {
                    index = (index + probe) & this.mask;
                }

                this.entries[index] = entry;
            }
        }

        private void Initialize()
        {
            const int InitSize = 4;
            
            this.entries = new Entry[InitSize];
            this.generation = 1;
            this.growThreshold = InitSize / 2;
            this.mask = InitSize - 1;
            this.filledEntriesCount = 0;
        }
    }
}
