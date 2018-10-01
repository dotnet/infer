// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.Graphs
{
    /// <summary>
    /// A list which is a union of labeled sublists.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <typeparam name="LabelType"></typeparam>
    /// <remarks>
    /// This class can be used to represent an inverted index, where each label maps to a list of items.
    /// It is implemented by a Dictionary of List objects, so the labels can have any type.
    /// </remarks>
    internal class LabeledList<T, LabelType> : LabeledCollection<T, LabelType>
    {
        private Dictionary<LabelType, ICollection<T>> dictionary;
        public LabelType defaultLabel;

        public override LabelType DefaultLabel
        {
            get { return defaultLabel; }
        }

        public LabeledList()
        {
            dictionary = new Dictionary<LabelType, ICollection<T>>();
        }

        public LabeledList(LabelType defaultLabel) : this()
        {
            this.defaultLabel = defaultLabel;
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="list"></param>
        public LabeledList(LabeledList<T, LabelType> list) : this()
        {
            // copy the elements into this list
            foreach (LabelType label in list.Labels)
            {
                foreach (T item in list.WithLabel(label))
                {
                    WithLabel(label).Add(item);
                }
            }
        }

        #region LabeledCollection methods

        public override ICollection<LabelType> Labels
        {
            get { return dictionary.Keys; }
        }

        public override ICollection<T> WithLabel(LabelType label)
        {
            ICollection<T> list;
            if (! dictionary.TryGetValue(label, out list))
            {
                // first time using this label
                list = new List<T>();
                dictionary[label] = list;
            }
            return list;
        }

        #endregion

        // this is more efficient than the default implementation
        public override void Clear()
        {
            dictionary.Clear();
        }
    }
}