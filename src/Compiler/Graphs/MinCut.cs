// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Reference: Jianxiu Hao and James B. Orlin, 
// "A faster algorithm for finding the minimum cut in a directed graph", 
// Journal of Algorithms 17: 424--446 (1994).

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Probabilistic.Collections;

namespace Microsoft.ML.Probabilistic.Compiler.Graphs
{
    /// <summary>
    /// Finds a minimum edge cut to separate a set of sources from a set of sinks
    /// </summary>
    /// <typeparam name="NodeType">Node type</typeparam>
    /// <typeparam name="EdgeType">Edge type</typeparam>
    /// <remarks><para>
    /// A node can be both a source and a sink.  In this case, edges into the node are considered toward the sink
    /// and edges out of the node are considered from the source.  Thus the algorithm will cut all paths from the
    /// node back to itself.
    /// </para><para>
    /// By modifying IsSinkEdge, certain edges can be labelled as "sink edges".  These edges are treated as if 
    /// their target was a sink node.  Making a node both a source and sink is equivalent to making it a source and
    /// labelling its inward edges as sink edges.
    /// </para><para>
    /// The implementation uses the preflow-push algorithm, modified to return the minimum cut,
    /// as described by Jianxiu Hao and James B. Orlin, 
    /// "A faster algorithm for finding the minimum cut in a directed graph", 
    /// Journal of Algorithms 17: 424--446 (1994).
    /// </para><para>
    /// This algorithm can sometimes be very slow for certain choices of the capacities.
    /// This seems to be caused by loss of precision in the float calculations, e.g.
    /// when a edge with capacity 1e-8 pushes flow into an edge with capacity 1e+8.
    /// </para></remarks>
    internal class MinCut<NodeType, EdgeType>
    {
        protected IDirectedGraph<NodeType, EdgeType> graph;
        protected Func<EdgeType, float> capacity;

        /// <summary>
        /// Capacity of an edge in the reverse direction (default 0)
        /// </summary>
        public Func<EdgeType, float> reverseCapacity;

        /// <summary>
        /// The set of source nodes
        /// </summary>
        public Set<NodeType> Sources = new Set<NodeType>();

        /// <summary>
        /// The set of sink nodes
        /// </summary>
        public Set<NodeType> Sinks = new Set<NodeType>();

        /// <summary>
        /// The set of sink edges
        /// </summary>
        public Func<EdgeType, bool> IsSinkEdge;

        private IndexedProperty<NodeType, int> distanceToSink; // also called "height"

        /// <summary>
        /// A cache of the nodes at a given distanceToSink
        /// </summary>
        private Dictionary<int, Set<NodeType>> nodesAtDistance = new Dictionary<int, Set<NodeType>>();

        /// <summary>
        /// The flow in the direction of the edge (always between -reverseCapacity and the capacity)
        /// </summary>
        private IndexedProperty<EdgeType, float> flow;

        /// <summary>
        /// A cache of (inward flow - outward flow)
        /// </summary>
        private IndexedProperty<NodeType, float> excess;

        /// <summary>
        /// The nodes on the source side of the cut
        /// </summary>
        private Set<NodeType> sourceGroup = new Set<NodeType>();

        /// <summary>
        /// All nodes that are not in the sourceGroup, not a sink, and have inward flow > outward flow
        /// </summary>
        private Set<NodeType> activeNodes = new Set<NodeType>();

        public MinCut(IDirectedGraph<NodeType, EdgeType> graph, Func<EdgeType, float> capacity)
        {
            this.graph = graph;
            this.capacity = capacity;
            reverseCapacity = e => 0f;
            distanceToSink = ((CanCreateNodeData<NodeType>) graph).CreateNodeData<int>(1);
            flow = ((CanCreateEdgeData<EdgeType>) graph).CreateEdgeData<float>(0f);
            excess = ((CanCreateNodeData<NodeType>) graph).CreateNodeData<float>(0f);
            IsSinkEdge = e => false;
        }

        private void Initialize()
        {
            distanceToSink.Clear();
            flow.Clear();
            excess.Clear();
            activeNodes.Clear();
            sourceGroup.Clear();
            sourceGroup.AddRange(Sources);
            nodesAtDistance[0] = Set<NodeType>.FromEnumerable(Sinks);
            foreach (NodeType sink in Sinks) distanceToSink[sink] = 0;
            Set<NodeType> nodesAtDistance1 = new Set<NodeType>();
            foreach (NodeType node in graph.Nodes)
            {
                if (!Sources.Contains(node) && !Sinks.Contains(node)) nodesAtDistance1.Add(node);
            }
            nodesAtDistance[1] = nodesAtDistance1;
            foreach (NodeType source in Sources)
            {
                if (!Sinks.Contains(source)) distanceToSink[source] = int.MaxValue;
                foreach (EdgeType edge in graph.EdgesOutOf(source))
                {
                    float f = capacity(edge);
                    flow[edge] = f;
                    if (!IsSinkEdge(edge))
                    {
                        NodeType target = graph.TargetOf(edge);
                        if (!Sources.Contains(target) && !Sinks.Contains(target))
                        {
                            excess[target] += f;
                            activeNodes.Add(target);
                        }
                    }
                }
                foreach (EdgeType edge in graph.EdgesInto(source))
                {
                    if (!IsSinkEdge(edge))
                    {
                        float f = reverseCapacity(edge);
                        flow[edge] = -f;
                        NodeType target = graph.SourceOf(edge);
                        if (!Sources.Contains(target) && !Sinks.Contains(target))
                        {
                            excess[target] += f;
                            activeNodes.Add(target);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Compute the min cut and return all nodes connected to any source
        /// </summary>
        /// <returns>The set of all nodes connected to any source after removing the min cut edges</returns>
        public Set<NodeType> GetSourceGroup()
        {
            Initialize();
            while (activeNodes.Count > 0)
            {
                //Console.WriteLine(StringUtil.CollectionToString(activeNodes," "));
                // select an active node
                NodeType node = activeNodes.First();
                if (float.IsNaN(excess[node])) throw new Exception("encountered NaN");
                if (excess[node] > 0f)
                {
                    // discharge this node
                    // find an admissible edge
                    // an edge is admissible if neither endpoint is in the sourceGroup (unless target is a sink), distanceToSink(source)=distanceToSink(target)+1, and residual capacity>0
                    foreach (EdgeType edge in graph.EdgesOutOf(node))
                    {
                        NodeType target = graph.TargetOf(edge);
                        bool isSinkEdge = IsSinkEdge(edge);
                        int distTarget = isSinkEdge ? 0 : distanceToSink[target];
                        if (distanceToSink[node] - 1 != distTarget) continue;
                        // target can be in sourceGroup if it is a sink
                        float cap = capacity(edge);
                        float residual;
                        if (cap == flow[edge]) residual = 0f; // avoid infinity-infinity
                        else residual = cap - flow[edge];
                        if (residual <= 0f) continue;
                        // push flow along this edge
                        float push = System.Math.Min(excess[node], residual);
                        flow[edge] += push;
                        if (excess[node] == push) excess[node] = 0f; // avoid infinity-infinity
                        else excess[node] -= push;
                        if (!isSinkEdge && !Sinks.Contains(target))
                        {
                            excess[target] += push;
                            if (excess[target] > 0f) activeNodes.Add(target);
                        }
                        if (excess[node] <= 0f) break;
                    }
                }
                if (excess[node] > 0f)
                {
                    // push flow back to inward edges
                    foreach (EdgeType edge in graph.EdgesInto(node))
                    {
                        NodeType target = graph.SourceOf(edge);
                        if (distanceToSink[node] - 1 != distanceToSink[target]) continue;
                        // if target is both a source and sink, its distanceToSink will be zero, but we cannot push flow backward into it
                        if (Sources.Contains(target)) continue;
                        float revCap = reverseCapacity(edge);
                        float residual;
                        if (-revCap == flow[edge]) residual = 0f;
                        else residual = revCap + flow[edge];
                        if (residual <= 0f) continue;
                        // push flow backward along this edge
                        float push = System.Math.Min(excess[node], residual);
                        if (flow[edge] == push) flow[edge] = 0f;
                        else flow[edge] -= push;
                        if (excess[node] == push) excess[node] = 0f;
                        else excess[node] -= push;
                        excess[target] += push;
                        if (!Sinks.Contains(target) && excess[target] > 0f) activeNodes.Add(target);
                        if (excess[node] <= 0f) break;
                    }
                }
                if (excess[node] > 0f)
                {
                    // relabel
                    int dist = distanceToSink[node];
                    int count = nodesAtDistance[dist].Count;
                    if (count == 1)
                    {
                        // cut at this node
                        foreach (KeyValuePair<int, Set<NodeType>> entry in nodesAtDistance)
                        {
                            if (entry.Key >= dist)
                            {
                                foreach (NodeType node2 in entry.Value)
                                {
                                    sourceGroup.Add(node2);
                                    distanceToSink[node2] = int.MaxValue;
                                    activeNodes.Remove(node2);
                                }
                                entry.Value.Clear();
                            }
                        }
                    }
                    else
                    {
                        // update the distance to sink
                        nodesAtDistance[dist].Remove(node);
                        dist = int.MaxValue;
                        foreach (EdgeType edge in graph.EdgesOutOf(node))
                        {
                            NodeType target = graph.TargetOf(edge);
                            if (sourceGroup.Contains(target)) continue;
                            float cap = capacity(edge);
                            float residual;
                            if (cap == flow[edge]) residual = 0f; // for infinity case
                            else residual = cap - flow[edge];
                            if (residual <= 0f) continue;
                            int distTarget = IsSinkEdge(edge) ? 0 : distanceToSink[target];
                            dist = System.Math.Min(dist, distTarget + 1);
                        }
                        foreach (EdgeType edge in graph.EdgesInto(node))
                        {
                            NodeType target = graph.SourceOf(edge);
                            if (sourceGroup.Contains(target)) continue;
                            float revCap = reverseCapacity(edge);
                            float residual;
                            if (-revCap == flow[edge]) residual = 0f;
                            else residual = revCap + flow[edge];
                            if (residual <= 0f) continue;
                            dist = System.Math.Min(dist, distanceToSink[target] + 1);
                        }
                        distanceToSink[node] = dist;
                        if (dist == int.MaxValue)
                        {
                            sourceGroup.Add(node);
                            // node is no longer active
                        }
                        else
                        {
                            Set<NodeType> nodes;
                            if (!nodesAtDistance.TryGetValue(dist, out nodes))
                            {
                                nodes = new Set<NodeType>();
                                nodesAtDistance[dist] = nodes;
                            }
                            nodes.Add(node);
                            continue; // node remains active
                        }
                    }
                }
                activeNodes.Remove(node);
            }
            return sourceGroup;
        }
    }
}