// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler.Graphs;
using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.Graphs
{
    /// <summary>
    /// A set which is the union of labeled subsets.
    /// </summary>
    /// <typeparam name="ItemType"></typeparam>
    /// <typeparam name="LabelType"></typeparam>
    /// <remarks>
    /// It is implemented as a LabeledList where Add is overridden to prevent adding duplicates.
    /// </remarks>
    internal class LabeledSet<ItemType, LabelType> : LabeledSetWrapper<ItemType, LabelType, LabeledList<ItemType, LabelType>>
    {
        public LabeledSet()
            : base(new LabeledList<ItemType, LabelType>())
        {
        }

        public LabeledSet(LabelType defaultLabel)
            : base(new LabeledList<ItemType, LabelType>(defaultLabel))
        {
        }
    }

    internal class LabeledSetWrapper<NodeType, LabelType, ListType> : LabeledCollectionWrapper<NodeType, LabelType, ListType>
        where ListType : ILabeledCollection<NodeType, LabelType>
    {
        public LabeledSetWrapper(ListType list)
            : base(list, null)
        {
            factory = delegate(ICollection<NodeType> sublist) { return new NodeListWrapper(sublist, this); };
        }

        // These routines are adapted from NodeListWrapper

        #region ICollection methods

        // value can be a duplicate node, but it won't be added again.
        public override void Add(NodeType node)
        {
            if (!list.Contains(node)) list.Add(node);
        }

        #endregion

        // Wraps a node list to ensure that nodes are unique, and that
        // graph edges are removed when a node is removed.
        private class NodeListWrapper : CollectionWrapper<NodeType, ICollection<NodeType>>
        {
            private LabeledSetWrapper<NodeType, LabelType, ListType> set;

            public NodeListWrapper(ICollection<NodeType> list, LabeledSetWrapper<NodeType, LabelType, ListType> set)
                : base(list)
            {
                this.set = set;
            }

            #region ICollection methods

            // value can be a duplicate node, but it won't be added again.
            public override void Add(NodeType item)
            {
                if (!set.Contains(item)) list.Add(item);
            }

            #endregion
        }
    }
}