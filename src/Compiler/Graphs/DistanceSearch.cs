// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.Graphs
{
    /// <summary>
    /// Computes the distance to all nodes reachable from a starting node.
    /// </summary>
    /// <typeparam name="NodeType">The node type.</typeparam>
    /// <remarks>
    /// The distances are returned via the SetDistance action.  Nodes which are unreachable from the
    /// starting node will have no distance set.
    /// </remarks>
    internal class DistanceSearch<NodeType>
    {
        protected Converter<NodeType, IEnumerable<NodeType>> Successors;
        protected CanCreateNodeData<NodeType> Data;
        public event Action<NodeType, int> SetDistance;
        private readonly BreadthFirstSearch<NodeType> bfs;
        private int parentCount, childCount, distance;

        public DistanceSearch(IGraph<NodeType> graph)
            : this(graph.NeighborsOf, (CanCreateNodeData<NodeType>) graph)
        {
        }

        public DistanceSearch(IDirectedGraph<NodeType> graph)
            : this(graph.TargetsOf, (CanCreateNodeData<NodeType>) graph)
        {
        }

        public DistanceSearch(Converter<NodeType, IEnumerable<NodeType>> successors,
                              CanCreateNodeData<NodeType> data)
        {
            this.Successors = successors;
            this.Data = data;
            bfs = new BreadthFirstSearch<NodeType>(Successors, Data);
            bfs.DiscoverNode += delegate(NodeType node)
                {
                    OnSetDistance(node, distance);
                    if (distance == 0) distance++;
                    else childCount++;
                };
            bfs.FinishNode += delegate(NodeType node)
                {
                    if (--parentCount == 0)
                    {
                        parentCount = childCount;
                        childCount = 0;
                        distance++;
                    }
                };
        }

        public void SearchFrom(NodeType start)
        {
            // we can compute the distance from the starting node to any node using constant additional storage.
            // at any given time, the bfs queue contains a set of parent nodes whose distance is <c>distance</c>
            // and a set of child nodes whose distance is <c>distance+1</c>.  When the parent nodes are
            // exhausted, the children become parents and we reset childCount to 0, incrementing distance.
            parentCount = 1;
            childCount = 0;
            distance = 0;
            bfs.SearchFrom(start);
            bfs.Clear();
        }

        public void OnSetDistance(NodeType node, int distance)
        {
            SetDistance?.Invoke(node, distance);
        }
    }
}