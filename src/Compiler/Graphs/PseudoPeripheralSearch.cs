// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.Graphs
{
    /// <summary>
    /// Searches for a pair of pseudo-peripheral nodes in a graph.
    /// </summary>
    /// <remarks>
    /// The nodes <c>(start,end)</c> form a pseudo-peripheral pair
    /// if <c>end</c> is the furthest node from <c>start</c> and <c>start</c> is the
    /// furthest node from <c>end</c>.  
    /// If the distance is maximal over all such pairs, then
    /// the nodes are peripheral. 
    /// This class does not guarantee that the pair is peripheral, only pseudo-peripheral.
    /// In a directed graph, distance from <c>start</c> is measured forward and distance from
    /// <c>end</c> is measured backward.
    /// </remarks>
    internal class PseudoPeripheralSearch<NodeType>
    {
        private Converter<NodeType, IEnumerable<NodeType>> Successors;
        private Converter<NodeType, IEnumerable<NodeType>> Predecessors;
        private CanCreateNodeData<NodeType> Data;

        public PseudoPeripheralSearch(IGraph<NodeType> graph)
            : this(graph.NeighborsOf, graph.NeighborsOf, (CanCreateNodeData<NodeType>) graph)
        {
        }

        public PseudoPeripheralSearch(IDirectedGraph<NodeType> graph)
            : this(graph.TargetsOf, graph.SourcesOf, (CanCreateNodeData<NodeType>) graph)
        {
        }

        public PseudoPeripheralSearch(Converter<NodeType, IEnumerable<NodeType>> successors,
                                      Converter<NodeType, IEnumerable<NodeType>> predecessors,
                                      CanCreateNodeData<NodeType> data)
        {
            this.Successors = successors;
            this.Predecessors = predecessors;
            this.Data = data;
        }

        /// <summary>
        /// Find a pseudo-peripheral pair.
        /// </summary>
        /// <param name="start">On entry, holds an initial seed for the search.  On return, holds the start node of the pair.</param>
        /// <param name="end">On return, holds the end node of the pair.</param>
        /// <remarks>Regardless of the seed node provided, a pseudo-peripheral pair will always be found.
        /// However the seed can affect the quality of the pair (i.e. how distant they are).</remarks>
        public void SearchFrom(ref NodeType start, out NodeType end)
        {
            NodeType startNode = start, endNode = start;
            int maxDistance = 0;
            bool firstTime = true;
            // this loop will always terminate because maxDistance only increases.
            while (true)
            {
                DistanceSearch<NodeType> distanceForward = new DistanceSearch<NodeType>(Successors, Data);
                bool endMoved = false;
                distanceForward.SetDistance += delegate(NodeType node, int distance)
                    {
                        //Console.WriteLine("forward: "+node+distance);
                        if (distance > maxDistance)
                        {
                            maxDistance = distance;
                            endNode = node;
                            endMoved = true;
                        }
                    };
                distanceForward.SearchFrom(startNode);
                if (firstTime) firstTime = false;
                else if (!endMoved) break;
                distanceForward = null;

                DistanceSearch<NodeType> distanceBackward = new DistanceSearch<NodeType>(Predecessors, Data);
                bool startMoved = false;
                distanceBackward.SetDistance += delegate(NodeType node, int distance)
                    {
                        //Console.WriteLine("backward: "+node+distance);
                        if (distance > maxDistance)
                        {
                            maxDistance = distance;
                            startNode = node;
                            startMoved = true;
                        }
                    };
                distanceBackward.SearchFrom(endNode);
                if (!startMoved) break;
                distanceBackward = null;
            }
            start = startNode;
            end = endNode;
        }
    }
}