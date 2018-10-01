// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Collections
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;

    /// <summary>
    /// A pool of items that can be registered against a given key.
    /// </summary>
    /// <typeparam name="TKey">The type of a key.</typeparam>
    /// <typeparam name="TItem">The type of an item.</typeparam>
    public class KeyedPool<TKey, TItem>
    {
        /// <summary>
        /// A mapping from a key to a list of items registered against that key.
        /// </summary>
        private readonly Dictionary<TKey, LinkedList<TItem>> storage = new Dictionary<TKey, LinkedList<TItem>>();

        /// <summary>
        /// A factory for producing new items.
        /// </summary>
        private readonly Func<TItem> itemFactory;

        /// <summary>
        /// Initializes a new instance of the <see cref="KeyedPool{TKey,TItem}"/> class.
        /// </summary>
        /// <param name="itemFactory">A factory for producing new items.</param>
        public KeyedPool(Func<TItem> itemFactory)
        {
            this.itemFactory = itemFactory;
        }

        /// <summary>
        /// Returns an item registered against a given key. If there are no items registered against that key,
        /// returns an item registered against any key. If there are no registered items at all,
        /// returns a new one.
        /// </summary>
        /// <param name="desiredKey">The desired key.</param>
        /// <returns>An item.</returns>
        public TItem Acquire(TKey desiredKey)
        {
            // Might not be the most efficient way to implement it,
            // but this class isn't meant to be used in a highly concurrent environment.
            lock (this.storage)
            {
                if (this.storage.Count > 0)
                {
                    // Try to get an item for the key
                    LinkedList<TItem> values;
                    if (this.storage.TryGetValue(desiredKey, out values) && values.Count > 0)
                    {
                        return this.ExtractItem(desiredKey, values);
                    }

                    // Return any item
                    var anyRecord = this.storage.First();
                    return this.ExtractItem(anyRecord.Key, anyRecord.Value);
                }
            }

            // Create a new item
            return this.itemFactory();
        }

        /// <summary>
        /// Returns a given item to the pool and registers it against a provided key.
        /// </summary>
        /// <param name="item">The item.</param>
        /// <param name="key">The key to register the item against.</param>
        public void Release(TItem item, TKey key)
        {
            lock (this.storage)
            {
                LinkedList<TItem> items;
                if (!this.storage.TryGetValue(key, out items))
                {
                    items = new LinkedList<TItem>();
                    this.storage.Add(key, items);
                }

                items.AddFirst(item);
            }
        }

        /// <summary>
        /// Extracts an item from the list of items registered against a given key.
        /// </summary>
        /// <param name="key">The key.</param>
        /// <param name="items">The list of items registered against <paramref name="key"/>.</param>
        /// <returns>The extracted item.</returns>
        private TItem ExtractItem(TKey key, LinkedList<TItem> items)
        {
            Debug.Assert(items.Count > 0, "The list of items must not be empty.");

            TItem result = items.First.Value;
            items.RemoveFirst();

            if (items.Count == 0)
            {
                // No more items? remove the key!
                this.storage.Remove(key);
            }

            return result;
        }
    }
}
