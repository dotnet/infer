// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler.Graphs;
using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.Graphs
{
    internal delegate ICollection<T> CollectionWrapperFactory<T>(ICollection<T> list);

    /// <summary>
    /// A base class for LabeledCollection wrapper classes.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <typeparam name="LabelType"></typeparam>
    /// <typeparam name="ListType"></typeparam>
    /// <remarks>
    /// This class makes it easy to write decorators for LabeledCollections.
    /// </remarks>
    internal class LabeledCollectionWrapper<T, LabelType, ListType> : CollectionWrapper<T, ListType>, ILabeledCollection<T, LabelType>
        where ListType : ILabeledCollection<T, LabelType>
    {
        protected CollectionWrapperFactory<T> factory;

        protected LabeledCollectionWrapper()
        {
        }

        public LabeledCollectionWrapper(ListType list, CollectionWrapperFactory<T> factory)
            : base(list)
        {
            this.factory = factory;
        }

        #region ILabeledCollection methods

        public ICollection<LabelType> Labels
        {
            get { return list.Labels; }
        }

        public ICollection<T> WithLabel(LabelType label)
        {
            return factory(list.WithLabel(label));
        }

        #endregion
    }
}