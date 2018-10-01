// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.Graphs
{
    internal struct IndexedProperty<KeyType, ValueType>
    {
        /// <summary>
        /// Delegate for retrieving data at an index.
        /// </summary>
        public Converter<KeyType, ValueType> Get;

        /// <summary>
        /// Delegate for setting data at an index.
        /// </summary>
        public Action<KeyType, ValueType> Set;

        /// <summary>
        /// Delegate for clearing the mapping.
        /// </summary>
        /// <remarks>
        /// If the mapping has a default value, this sets all data to that value.
        /// Otherwise the mapping is undefined at all values.
        /// </remarks>
        public Action Clear;

        /// <summary>
        /// Get or set data at an index.
        /// </summary>
        /// <param name="key"></param>
        /// <returns></returns>
        public ValueType this[KeyType key]
        {
            get { return Get(key); }
            set { Set(key, value); }
        }

        public IndexedProperty(Converter<KeyType, ValueType> getter, Action<KeyType, ValueType> setter, Action clearer)
        {
            Get = getter;
            Set = setter;
            Clear = clearer;
        }

        public IndexedProperty(IDictionary<KeyType, ValueType> dictionary, ValueType defaultValue = default(ValueType))
        {
            Get = delegate(KeyType key)
                {
                    ValueType value;
                    bool containsKey = dictionary.TryGetValue(key, out value);
                    if (!containsKey) return defaultValue;
                    else return value;
                };
            Set = delegate(KeyType key, ValueType value) { dictionary[key] = value; };
            Clear = dictionary.Clear;
        }
    }

    internal static class MakeIndexedProperty
    {
        public static IndexedProperty<int, T> FromArray<T>(T[] array, T defaultValue = default(T))
        {
            return new IndexedProperty<int, T>(
                delegate(int key) { return array[key]; },
                delegate(int key, T value) { array[key] = value; },
                delegate()
                    {
                        if (ReferenceEquals(defaultValue, null) || defaultValue.Equals(default(T)))
                            Array.Clear(array, 0, array.Length);
                        else
                        {
                            for (int i = 0; i < array.Length; i++)
                            {
                                array[i] = defaultValue;
                            }
                        }
                    });
        }

        public static IndexedProperty<T, bool> FromSet<T>(ICollection<T> set)
        {
            return new IndexedProperty<T, bool>(
                delegate (T key)
                { return set.Contains(key); },
                delegate (T key, bool value)
                {
                    if (value)
                        set.Add(key);
                    else
                        set.Remove(key);
                },
                delegate ()
                {
                    set.Clear();
                });
        }
    }
}