// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions.Automata
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// Contains methods for converting automata into regular expressions.
    /// </summary>
    public static class RegexpTreeBuilder
    {
        /// <summary>
        /// Type used to track edge status in graph operations for building regular expressions.
        /// </summary>
        [Flags]
        private enum EdgeTypes
        {
            /// <summary>
            /// An edge in the original graph constructed from the automaton.
            /// </summary>
            Original = 0x01,

            /// <summary>
            /// An edge formed by collapsing two edges in series into a single edge.
            /// </summary>
            Chain = 0x02,

            /// <summary>
            /// An edge added by the transitive closure algorithm.
            /// </summary>
            AddedByTC = 0x04,

            /// <summary>
            /// An original or chain edge removed by the transitive reduction algorithm.
            /// </summary>
            RemovedByTR = 0x08,

            /// <summary>
            /// An edge that has been collapsed by considering nested intervals.
            /// </summary>
            CollapsedInterval = 0x10
        }
        
        /// <summary>
        /// Builds a regular expression tree for the support of a given automaton.
        /// </summary>
        /// <typeparam name="TSequence">The type of a sequence.</typeparam>
        /// <typeparam name="TElement">The type of a sequence element.</typeparam>
        /// <typeparam name="TElementDistribution">The type of a distribution over sequence elements.</typeparam>
        /// <typeparam name="TSequenceManipulator">The type providing ways to manipulate sequences.</typeparam>
        /// <typeparam name="TAutomaton">The concrete type of an automaton.</typeparam>
        /// <param name="automaton">The automaton.</param>
        /// <param name="collapseAlternatives">
        /// Specifies whether an attempt to merge identical sub-expressions should be made.
        /// Setting it to <see langword="false"/> will improve the performance, but produce longer regular expressions.
        /// Defaults to <see langword="true"/>.
        /// </param>
        /// <returns>The built regular expression tree.</returns>
        public static RegexpTreeNode<TElement> BuildRegexp<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>(
            Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton> automaton,
            bool collapseAlternatives = true)
            where TSequence : class, IEnumerable<TElement>
            where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>,
                SettableToPartialUniform<TElementDistribution>, new()
            where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
            where TAutomaton : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>, new()
        {
            bool buildCompactRegex = true;

            if (buildCompactRegex)
            {
                // BuildCompactRegex is the more recent implementation. The old implementation is maintained
                // for now, but is unused.
                return BuildCompactRegex(automaton, collapseAlternatives);
            }

            var condensation = automaton.ComputeCondensation(automaton.Start);

            RegexpTreeNode<TElement>[][,] componentRegexps = Util.ArrayInit(
                condensation.ComponentCount, i => BuildStronglyConnectedComponentRegexp(condensation.GetComponent(i)));
            RegexpTreeNode<TElement>[] stateRegexps = Util.ArrayInit(
                automaton.States.Count, i => RegexpTreeNode<TElement>.Nothing());

            // Dynamic programming in reverse topological order
            for (int componentIndex = 0; componentIndex < condensation.ComponentCount; ++componentIndex)
            {
                var currentComponent = condensation.GetComponent(componentIndex);
                for (int stateIndex = 0; stateIndex < currentComponent.Size; ++stateIndex)
                {
                    var state = currentComponent.GetStateByIndex(stateIndex);
                    RegexpTreeNode<TElement> stateDownwardRegexp = state.CanEnd ? RegexpTreeNode<TElement>.Empty() : RegexpTreeNode<TElement>.Nothing();
                    for (int transitionIndex = 0; transitionIndex < state.TransitionCount; ++transitionIndex)
                    {
                        var transition = state.GetTransition(transitionIndex);
                        var destState = automaton.States[transition.DestinationStateIndex];
                        if (!transition.Weight.IsZero && !currentComponent.HasState(destState))
                        {
                            var destStateRegexp = transition.IsEpsilon
                                ? RegexpTreeNode<TElement>.Empty()
                                : RegexpTreeNode<TElement>.FromElementSet(transition.ElementDistribution);
                            stateDownwardRegexp = RegexpTreeNode<TElement>.Or(
                                stateDownwardRegexp,
                                RegexpTreeNode<TElement>.Concat(destStateRegexp, stateRegexps[transition.DestinationStateIndex]));
                        }
                    }

                    for (int updatedStateIndex = 0; updatedStateIndex < currentComponent.Size; ++updatedStateIndex)
                    {
                        var updatedState = currentComponent.GetStateByIndex(updatedStateIndex);
                        stateRegexps[updatedState.Index] = RegexpTreeNode<TElement>.Or(
                            stateRegexps[updatedState.Index],
                            RegexpTreeNode<TElement>.Concat(componentRegexps[componentIndex][updatedStateIndex, stateIndex], stateDownwardRegexp));
                    }
                }
            }

            RegexpTreeNode<TElement> result = stateRegexps[automaton.Start.Index];
            result.Simplify(collapseAlternatives);

            return result;
        }

        /// <summary>
        /// Builds a regular expression tree for the support of a given automaton.
        /// </summary>
        /// <typeparam name="TSequence">The type of a sequence.</typeparam>
        /// <typeparam name="TElement">The type of a sequence element.</typeparam>
        /// <typeparam name="TElementDistribution">The type of a distribution over sequence elements.</typeparam>
        /// <typeparam name="TSequenceManipulator">The type providing ways to manipulate sequences.</typeparam>
        /// <typeparam name="TAutomaton">The concrete type of an automaton.</typeparam>
        /// <param name="automaton">The automaton.</param>
        /// <param name="collapseAlternatives">
        /// Specifies whether an attempt to merge identical sub-expressions should be made.
        /// Setting it to <see langword="false"/> will improve the performance, but produce longer regular expressions.
        /// Defaults to <see langword="true"/>.
        /// </param>
        /// <returns>The built regular expression tree.</returns>
        public static RegexpTreeNode<TElement> BuildCompactRegex<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>(
            Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton> automaton,
            bool collapseAlternatives = true)
            where TSequence : class, IEnumerable<TElement>
            where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>,
                SettableToPartialUniform<TElementDistribution>, new()
            where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
            where TAutomaton : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>, new()
        {
            const int EndNodeIndex = 0;
            List<int> sizeOneList = new List<int>() { 0 };

            // Create the graph
            var condensation = automaton.ComputeCondensation(automaton.Start);

            // The extra node is the end node.
            var graphNodes = new List<GraphNode<TElement>>();
            var stateIndexToNodeIndex = new Dictionary<int, int>();

            int nodeIndex = 0;
            graphNodes.Add(new GraphNode<TElement>(nodeIndex));

            for (int componentIndex = 0; componentIndex < condensation.ComponentCount; ++componentIndex)
            {
                var currentComponent = condensation.GetComponent(componentIndex);
                var componentRegexps = BuildStronglyConnectedComponentRegexp(condensation.GetComponent(componentIndex));

                var stateIndicesWithOutgoingEdges =
                    currentComponent.Size == 1 ?
                    sizeOneList :
                    Enumerable.Range(0, currentComponent.Size).Where(i =>
                        {
                            var s = currentComponent.GetStateByIndex(i);
                            if (s.CanEnd)
                            {
                                return true;
                            }

                            for (int tidx = 0; tidx < s.TransitionCount; ++tidx)
                            {
                                var t = s.GetTransition(tidx);
                                var dest = automaton.States[t.DestinationStateIndex];
                                if ((!t.Weight.IsZero) && (!currentComponent.HasState(dest)))
                                {
                                    return true;
                                }
                            }

                            return false;
                        }).ToList();

                var stateIndicesWithIncomingEdges =
                    currentComponent.Size == 1 ?
                    sizeOneList :
                    Enumerable.Range(0, currentComponent.Size).Where(i =>
                        {
                            var s = currentComponent.GetStateByIndex(i);
                            var sIdx = s.Index;
                            foreach (var s1 in s.Owner.States)
                            {
                                if (currentComponent.HasState(s1))
                                {
                                    continue;
                                }

                                var trans = s1.GetTransitions();
                                for (int tIdx = 0; tIdx < trans.Length; tIdx++)
                                {
                                    var t = trans[tIdx];
                                    if (t.DestinationStateIndex == sIdx && !t.Weight.IsZero)
                                    {
                                        return true;
                                    }
                                }
                            }

                            return false;
                        }).ToList();

                var incomingAndOutgoingStateIndices = stateIndicesWithOutgoingEdges.Intersect(stateIndicesWithIncomingEdges).ToList();

                // If there is only one incoming node intersecting with outgoing nodes we can build edges internal to the component
                bool buildInternalEdges = incomingAndOutgoingStateIndices.Count <= 1;
                var allOutgoingRegexes = new Dictionary<int, Dictionary<int, RegexpTreeNode<TElement>>>();
                GraphNode<TElement> currentNode = null;
                foreach (int stateIndex in stateIndicesWithOutgoingEdges)
                {
                    var state = currentComponent.GetStateByIndex(stateIndex);
                    if (buildInternalEdges)
                    {
                        stateIndexToNodeIndex[state.Index] = ++nodeIndex;
                        currentNode = new GraphNode<TElement>(nodeIndex);
                        currentNode.Regex = componentRegexps[stateIndex, stateIndex];
                    }

                    var outgoingRegexes = new Dictionary<int, RegexpTreeNode<TElement>>();
                    for (int transitionIndex = 0; transitionIndex < state.TransitionCount; ++transitionIndex)
                    {
                        var transition = state.GetTransition(transitionIndex);
                        var destState = automaton.States[transition.DestinationStateIndex];
                        if ((!transition.Weight.IsZero) && (!currentComponent.HasState(destState)))
                        {
                            var regex =
                                transition.IsEpsilon ?
                                RegexpTreeNode<TElement>.Empty() :
                                RegexpTreeNode<TElement>.FromElementSet(transition.ElementDistribution);

                            var destinationNodeIndex = stateIndexToNodeIndex[transition.DestinationStateIndex];
                            if (outgoingRegexes.ContainsKey(destinationNodeIndex))
                            {
                                outgoingRegexes[destinationNodeIndex] = RegexpTreeNode<TElement>.Or(outgoingRegexes[destinationNodeIndex], regex);
                            }
                            else
                            {
                                outgoingRegexes[destinationNodeIndex] = regex;
                            }
                        }
                    }

                    if (state.CanEnd)
                    {
                        outgoingRegexes[EndNodeIndex] = RegexpTreeNode<TElement>.Empty();
                    }

                    allOutgoingRegexes[stateIndex] = outgoingRegexes;

                    if (buildInternalEdges)
                    {
                        foreach (var kvp in outgoingRegexes)
                        {
                            var edge = new Edge<TElement>(kvp.Key);
                            edge.Regex = kvp.Value;
                            currentNode.OutgoingNodes.Add(edge);
                        }

                        graphNodes.Add(currentNode);
                    }
                }

                foreach (int stateIndex in stateIndicesWithIncomingEdges)
                {
                    var state = currentComponent.GetStateByIndex(stateIndex);
                    if (!stateIndexToNodeIndex.ContainsKey(state.Index))
                    {
                        stateIndexToNodeIndex[state.Index] = ++nodeIndex;
                        graphNodes.Add(new GraphNode<TElement>(nodeIndex));
                    }

                    currentNode = graphNodes[stateIndexToNodeIndex[state.Index]];
                    currentNode.Regex = componentRegexps[stateIndex, stateIndex];

                    if (buildInternalEdges)
                    {
                        foreach (int stateIndex2 in stateIndicesWithOutgoingEdges)
                        {
                            if (stateIndex == stateIndex2)
                            {
                                continue;
                            }

                            var state2 = currentComponent.GetStateByIndex(stateIndex2);
                            var edge = new Edge<TElement>(stateIndexToNodeIndex[state2.Index]);
                            edge.Regex = componentRegexps[stateIndex, stateIndex2];
                            currentNode.OutgoingNodes.Add(edge);
                        }
                    }
                    else
                    {
                        var outgoingRegexes = new Dictionary<int, RegexpTreeNode<TElement>>();
                        foreach (int stateIndex2 in stateIndicesWithOutgoingEdges)
                        {
                            var regex = RegexpTreeNode<TElement>.Empty();
                            var state2 = currentComponent.GetStateByIndex(stateIndex2);
                            if (stateIndex != stateIndex2)
                            {
                                regex = RegexpTreeNode<TElement>.Concat(componentRegexps[stateIndex, stateIndex2], componentRegexps[stateIndex, stateIndex2]);
                            }

                            foreach (var outgoingRegex in allOutgoingRegexes[stateIndex2])
                            {
                                var chainedRegex = RegexpTreeNode<TElement>.Concat(regex, outgoingRegex.Value);
                                if (!outgoingRegexes.ContainsKey(outgoingRegex.Key))
                                {
                                    outgoingRegexes.Add(outgoingRegex.Key, chainedRegex);
                                }
                                else
                                {
                                    outgoingRegexes[outgoingRegex.Key] = RegexpTreeNode<TElement>.Or(outgoingRegexes[outgoingRegex.Key], chainedRegex);
                                }
                            }
                        }

                        foreach (var kvp in outgoingRegexes)
                        {
                            var edge = new Edge<TElement>(kvp.Key);
                            edge.Regex = kvp.Value;
                            currentNode.OutgoingNodes.Add(edge);
                        }
                    }
                }
            }

            var graph = new Graph<TElement>();
            graph.Nodes = graphNodes.ToArray();
            var result = graph.BuildRegex(collapseAlternatives);
            return result;
        }

        /// <summary>
        /// Computes the regular expression for each pair of states in a given strongly connected component
        /// by using the generalized Floyd's algorithm on the regular expression semiring
        /// as described in <a href="http://cs.stackexchange.com/questions/2016/how-to-convert-finite-automata-to-regular-expressions"/>.
        /// First state in the pair is treated as the start state, the second one - as the accepting state.
        /// </summary>
        /// <typeparam name="TSequence">The type of a sequence.</typeparam>
        /// <typeparam name="TElement">The type of a sequence element.</typeparam>
        /// <typeparam name="TElementDistribution">The type of a distribution over sequence elements.</typeparam>
        /// <typeparam name="TSequenceManipulator">The type providing ways to manipulate sequences.</typeparam>
        /// <typeparam name="TAutomaton">The concrete type of an automaton.</typeparam>
        /// <param name="component">The strongly connected component to compute the regular expressions for.</param>
        /// <returns>A table containing the computed regular expressions.</returns>
        private static RegexpTreeNode<TElement>[,] BuildStronglyConnectedComponentRegexp<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>(
            Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>.StronglyConnectedComponent component)
            where TSequence : class, IEnumerable<TElement>
            where TElementDistribution : IDistribution<TElement>, SettableToProduct<TElementDistribution>, SettableToWeightedSumExact<TElementDistribution>, CanGetLogAverageOf<TElementDistribution>,
                SettableToPartialUniform<TElementDistribution>, new()
            where TSequenceManipulator : ISequenceManipulator<TSequence, TElement>, new()
            where TAutomaton : Automaton<TSequence, TElement, TElementDistribution, TSequenceManipulator, TAutomaton>, new()
        {
            var regexps = Util.ArrayInit(component.Size, component.Size, (i, j) => RegexpTreeNode<TElement>.Nothing());
            for (int stateIndex = 0; stateIndex < component.Size; ++stateIndex)
            {
                var state = component.GetStateByIndex(stateIndex);
                regexps[stateIndex, stateIndex] = RegexpTreeNode<TElement>.Empty();

                for (int transitionIndex = 0; transitionIndex < state.TransitionCount; ++transitionIndex)
                {
                    var transition = state.GetTransition(transitionIndex);
                    if (transition.Weight.IsZero)
                    {
                        continue;
                    }

                    int destStateIndex;
                    if ((destStateIndex = component.GetIndexByState(state.Owner.States[transition.DestinationStateIndex])) != -1)
                    {
                        var destStateRegexp = transition.IsEpsilon
                                ? RegexpTreeNode<TElement>.Empty()
                                : RegexpTreeNode<TElement>.FromElementSet(transition.ElementDistribution);
                        regexps[stateIndex, destStateIndex] = RegexpTreeNode<TElement>.Or(
                            regexps[stateIndex, destStateIndex], destStateRegexp);
                    }
                }
            }

            for (int k = 0; k < component.Size; ++k)
            {
                RegexpTreeNode<TElement> star = RegexpTreeNode<TElement>.Star(regexps[k, k]);
                for (int i = 0; i < component.Size; ++i)
                {
                    if (i == k || regexps[i, k].Type == RegexpTreeNodeType.Nothing)
                    {
                        continue;
                    }

                    for (int j = 0; j < component.Size; ++j)
                    {
                        if (j == k || regexps[k, j].Type == RegexpTreeNodeType.Nothing)
                        {
                            continue;
                        }

                        RegexpTreeNode<TElement> option = RegexpTreeNode<TElement>.Concat(regexps[i, k], RegexpTreeNode<TElement>.Concat(star, regexps[k, j]));
                        regexps[i, j] = RegexpTreeNode<TElement>.Or(regexps[i, j], option);
                    }
                }

                for (int i = 0; i < component.Size; ++i)
                {
                    regexps[i, k] = RegexpTreeNode<TElement>.Or(regexps[i, k], star);
                    regexps[k, i] = RegexpTreeNode<TElement>.Or(regexps[k, i], star);
                }

                regexps[k, k] = star;
            }

            return regexps;
        }

        /// <summary>
        /// An edge in the graph used for building regular expressions.
        /// </summary>
        /// <typeparam name="TElement">The type of the element.</typeparam>
        private class Edge<TElement>
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="Edge{TElement}"/> class.
            /// </summary>
            /// <param name="index">The index of the node at the other end of this edge (either incoming or outgoing).</param>
            public Edge(int index)
                : this(index, EdgeTypes.Original)
            {
            }

            /// <summary>
            /// Initializes a new instance of the <see cref="Edge{TElement}"/> class.
            /// </summary>
            /// <param name="index">The index of the node at the other end of this edge (either incoming or outgoing).</param>
            /// <param name="edgeType">The type of the edge.</param>
            public Edge(int index, EdgeTypes edgeType)
            {
                this.Index = index;
                this.EdgeType = edgeType;
                this.Regex = RegexpTreeNode<TElement>.Empty();
            }

            /// <summary>
            /// Gets or sets the index of the node at the other end of this edge (either incoming or outgoing).
            /// </summary>
            public int Index
            {
                get;
                set;
            }

            /// <summary>
            /// Gets or sets the type of the edge.
            /// </summary>
            public EdgeTypes EdgeType
            {
                get;
                set;
            }

            /// <summary>
            /// Gets or sets the regular expression fragment for the edge.
            /// </summary>
            public RegexpTreeNode<TElement> Regex
            {
                get;
                set;
            }

            /// <summary>
            /// String representation of the instance.
            /// </summary>
            /// <returns>A string representation of the instance.</returns>
            public override string ToString()
            {
                return string.Format("{0} ({1}, {2})", this.Index, this.Regex.Type, this.EdgeType);
            }
        }

        /// <summary>
        /// A node in the graph used for building regular expressions.
        /// </summary>
        /// <typeparam name="TElement">The type of the element.</typeparam>
        private class GraphNode<TElement>
        {
            /// <summary>
            /// An instance of a comparer for comparing edges.
            /// </summary>
            private static GraphEdgeComparer edgeComparer = new GraphEdgeComparer();

            /// <summary>
            /// Initializes a new instance of the <see cref="GraphNode{TElement}"/> class.
            /// </summary>
            /// <param name="index">The index of the node.</param>
            public GraphNode(int index)
            {
                this.Index = index;
                this.IncomingNodes = new HashSet<Edge<TElement>>(edgeComparer);
                this.OutgoingNodes = new HashSet<Edge<TElement>>(edgeComparer);
                this.Regex = RegexpTreeNode<TElement>.Empty();
            }

            /// <summary>
            /// Gets or sets the set of incoming edges.
            /// </summary>
            public HashSet<Edge<TElement>> IncomingNodes
            {
                get;
                set;
            }

            /// <summary>
            /// Gets or sets the set of outgoing edges.
            /// </summary>
            public HashSet<Edge<TElement>> OutgoingNodes
            {
                get;
                set;
            }

            /// <summary>
            /// Gets or sets the index of the node.
            /// </summary>
            public int Index
            {
                get;
                set;
            }

            /// <summary>
            /// Gets or sets the star expression for the node.
            /// </summary>
            public RegexpTreeNode<TElement> Regex
            {
                get;
                set;
            }

            /// <summary>
            /// Gets the outgoing edge instance between this node and the node given by the destination index.
            /// </summary>
            /// <param name="destinationIndex">The index of the destination node.</param>
            /// <returns>The edge instance or null if no such edge exists.</returns>
            public Edge<TElement> GetOutgoingEdge(int destinationIndex)
            {
                var edge = new Edge<TElement>(destinationIndex);
                if (this.OutgoingNodes.Contains(edge))
                {
                    return this.OutgoingNodes.First(e => edgeComparer.Equals(e, edge));
                }

                return null;
            }

            /// <summary>
            /// Gets the incoming edge instance between the node given by the source index and this node.
            /// </summary>
            /// <param name="sourceIndex">The index of the source node.</param>
            /// <returns>The edge instance or null if no such edge exists.</returns>
            public Edge<TElement> GetIncomingEdge(int sourceIndex)
            {
                var edge = new Edge<TElement>(sourceIndex);
                if (this.IncomingNodes.Contains(edge))
                {
                    return this.IncomingNodes.First(e => edgeComparer.Equals(e, edge));
                }

                return null;
            }

            /// <summary>
            /// String representation of the instance.
            /// </summary>
            /// <returns>A string representation of the instance.</returns>
            public override string ToString()
            {
                return string.Format("{0}Out: {1}", this.Regex.Type != RegexpTreeNodeType.Empty ? "X" : string.Empty, Util.CollectionToString(this.OutgoingNodes));
            }

            /// <summary>
            /// A class that provides comparer functionality for comparing edges.
            /// </summary>
            public class GraphEdgeComparer : IEqualityComparer<Edge<TElement>>
            {
                /// <summary>
                /// Determines whether two edges are the same edge.
                /// </summary>
                /// <param name="x">The first edge.</param>
                /// <param name="y">The second edge.</param>
                /// <returns>True if the instances are equal, false otherwise.</returns>
                public bool Equals(Edge<TElement> x, Edge<TElement> y)
                {
                    return x.Index == y.Index;
                }

                /// <summary>
                /// Gets the hash code for the given edge.
                /// </summary>
                /// <param name="obj">The edge instance.</param>
                /// <returns>The hash code.</returns>
                public int GetHashCode(Edge<TElement> obj)
                {
                    return obj.Index.GetHashCode();
                }
            }
        }

        /// <summary>
        /// Graph used for building regular expressions.
        /// </summary>
        /// <typeparam name="TElement">The type of the element.</typeparam>
        private class Graph<TElement>
        {
            /// <summary>
            /// Backing field for <see cref="Nodes"/>.
            /// </summary>
            private GraphNode<TElement>[] nodes;

            /// <summary>
            /// Gets or sets the nodes in the graph.
            /// </summary>
            public GraphNode<TElement>[] Nodes
            {
                get
                {
                    return this.nodes;
                }

                set
                {
                    this.nodes = value;
                    this.BuildIncomingEdgesFromOutgoingEdges();
                }
            }

            /// <summary>
            /// Gets a value indicating the number of nodes.
            /// </summary>
            public int NodeCount
            {
                get
                {
                    return this.Nodes.Length;
                }
            }

            /// <summary>
            /// Builds the regex from the graph.
            /// </summary>
            /// <param name="collapseAlternatives">Whether to collapse alternatives or not.</param>
            /// <returns>A regex tree representing the graph.</returns>
            public RegexpTreeNode<TElement> BuildRegex(bool collapseAlternatives)
            {
                this.CollapseChainsAndParallels();
                this.TCAlgorithm();
                this.TRAlgorithm();
                var regex = this.BuildRegexFromTRGraph();
                regex.Simplify(collapseAlternatives);
                return regex;
            }

            /// <summary>
            /// Iteratively collapses chains and parallel edges where possible.
            /// </summary>
            private void CollapseChainsAndParallels()
            {
                bool iterate = true;

                while (iterate)
                {
                    // Collapse chains and any resulting parallel loops
                    iterate = false;
                    for (int k = 1; k < this.NodeCount - 1; k++)
                    {
                        var nodek = this.Nodes[k];
                        if (nodek.IncomingNodes.Count == 1 && nodek.OutgoingNodes.Count == 1)
                        {
                            iterate = true;
                            var ei = nodek.IncomingNodes.First();
                            var ej = nodek.OutgoingNodes.First();
                            int i = ei.Index;
                            int j = ej.Index;
                            var nodei = this.Nodes[i];
                            var nodej = this.Nodes[j];
                            var edgeik = nodei.GetOutgoingEdge(k);
                            var edgekj = nodej.GetIncomingEdge(k);
                            var regex = RegexpTreeNode<TElement>.Concat(edgeik.Regex, nodek.Regex);
                            regex = RegexpTreeNode<TElement>.Concat(regex, edgekj.Regex);
                            nodei.OutgoingNodes.Remove(edgeik);
                            nodej.IncomingNodes.Remove(edgekj);

                            nodek.IncomingNodes.Clear();
                            nodek.OutgoingNodes.Clear();

                            // Add the new edge between i and j, but if one already exists
                            // then update its regex
                            var newIEdgeToJ = new Edge<TElement>(j, EdgeTypes.Chain);
                            var newJEdgeFromI = new Edge<TElement>(i, EdgeTypes.Chain);

                            if (nodei.OutgoingNodes.Contains(newIEdgeToJ))
                            {
                                var existingIEdgeToJ = nodei.GetOutgoingEdge(j);
                                var existingJEdgeFromI = nodej.GetIncomingEdge(i);
                                regex = RegexpTreeNode<TElement>.Or(regex, existingIEdgeToJ.Regex);
                                nodei.OutgoingNodes.Remove(existingIEdgeToJ);
                                nodej.IncomingNodes.Remove(existingJEdgeFromI);
                            }

                            newIEdgeToJ.Regex = regex;
                            nodei.OutgoingNodes.Add(newIEdgeToJ);
                            newJEdgeFromI.Regex = regex;
                            nodej.IncomingNodes.Add(newJEdgeFromI);
                        }
                    }
                }
            }

            /// <summary>
            /// Transitive closure algorithm.
            /// </summary>
            private void TCAlgorithm()
            {
                var comparer = new GraphNode<TElement>.GraphEdgeComparer();

                // In topological order (graph is in reverse topological order)
                for (int k = this.NodeCount - 2; k >= 0; k--)
                {
                    var nodek = this.Nodes[k];
                    foreach (Edge<TElement> ej in nodek.OutgoingNodes)
                    {
                        var nodej = this.Nodes[ej.Index];
                        foreach (Edge<TElement> ei in nodek.IncomingNodes)
                        {
                            var nodei = this.Nodes[ei.Index];
                            nodei.OutgoingNodes.Add(new Edge<TElement>(ej.Index, EdgeTypes.AddedByTC));
                            nodej.IncomingNodes.Add(new Edge<TElement>(ei.Index, EdgeTypes.AddedByTC));
                        }
                    }
                }
            }

            /// <summary>
            /// Transitive reduction algorithm. Transitive closure must be called first.
            /// </summary>
            private void TRAlgorithm()
            {
                var comparer = new GraphNode<TElement>.GraphEdgeComparer();

                // In reverse topological order
                for (int k = 1; k < this.NodeCount - 1; k++)
                {
                    var nodek = this.Nodes[k];
                    foreach (var ej in nodek.OutgoingNodes)
                    {
                        var nodej = this.Nodes[ej.Index];
                        foreach (var ei in nodek.IncomingNodes)
                        {
                            var nodei = this.Nodes[ei.Index];
                            if (nodei.OutgoingNodes.Contains(new Edge<TElement>(ej.Index)))
                            {
                                var matchij = nodei.OutgoingNodes.First(e => e.Index == ej.Index);
                                nodei.OutgoingNodes.Remove(matchij);
                                if (matchij.EdgeType != EdgeTypes.AddedByTC)
                                {
                                    var newEdgeij = new Edge<TElement>(ej.Index, EdgeTypes.RemovedByTR);
                                    newEdgeij.Regex = matchij.Regex;
                                    nodei.OutgoingNodes.Add(newEdgeij);
                                }

                                var matchji = nodej.IncomingNodes.First(e => e.Index == ei.Index);
                                nodej.IncomingNodes.Remove(matchji);
                                if (matchji.EdgeType != EdgeTypes.AddedByTC)
                                {
                                    var newEdgeji = new Edge<TElement>(ei.Index, EdgeTypes.RemovedByTR);
                                    newEdgeji.Regex = matchji.Regex;
                                    nodej.IncomingNodes.Add(newEdgeji);
                                }
                            }
                        }
                    }
                }
            }

            /// <summary>
            /// Builds the regex tree from the transitively reduced graph.
            /// </summary>
            /// <returns>A regex tree.</returns>
            private RegexpTreeNode<TElement> BuildRegexFromTRGraph()
            {
                var transitivelyReducedEdges = this.Nodes.SelectMany(node => node.OutgoingNodes.Where(e => e.EdgeType == EdgeTypes.RemovedByTR).Select(e => Pair.Create(node.Index, e.Index))).ToList();
                if (transitivelyReducedEdges.Count > 0)
                {
                    // Build an interval tree based on containment.
                    transitivelyReducedEdges.Sort(new ContainmentIntervalComparer());
                    var root = new IntervalTreeNode()
                    {
                        Interval = Pair.Create(this.NodeCount - 1, 0)
                    };

                    var currNode = root;
                    foreach (var interval in transitivelyReducedEdges)
                    {
                        bool success = root.TryAddInterval(interval);
                    }

                    root.SortAndMergeChildren();

                    Pair<int, int> enclosingInterval;
                    this.TryCollapsingIntervalNode(root, out enclosingInterval);
                }

                return this.BuildRegexBetweenNodeIndices(this.NodeCount - 1, 0, true);
            }

            /// <summary>
            /// Recursively builds the regex tree for an interval tree node.
            /// </summary>
            /// <param name="node">The interval tree node.</param>
            /// <returns>The regex tree.</returns>
            private RegexpTreeNode<TElement> BuildRegexForIntervalNode(IntervalTreeNode node)
            {
                // Children should already be ordered.
                var children = node.Children;
                int intervalStart = node.Interval.First;
                int intervalEnd = node.Interval.Second;

                var result = RegexpTreeNode<TElement>.Empty();
                var endOfLast = node.Interval.First;

                foreach (var child in children)
                {
                    int startOfCurrent = child.Interval.First;
                    if (endOfLast != startOfCurrent)
                    {
                        result = RegexpTreeNode<TElement>.Concat(result, this.BuildRegexBetweenNodeIndices(endOfLast, startOfCurrent, true));
                    }

                    result = RegexpTreeNode<TElement>.Concat(result, this.BuildRegexForIntervalNode(child));
                    endOfLast = child.Interval.Second;
                }

                if (intervalEnd != endOfLast)
                {
                    result = RegexpTreeNode<TElement>.Concat(result, this.BuildRegexBetweenNodeIndices(endOfLast, intervalEnd, true));
                }

                return result;
            }

            /// <summary>
            /// Tries to collapse an interval tree node.
            /// </summary>
            /// <param name="node">The interval tree node.</param>
            /// <param name="enclosingInterval">The enclosing interval (output).</param>
            /// <returns>True if successful, false otherwise.</returns>
            private bool TryCollapsingIntervalNode(IntervalTreeNode node, out Pair<int, int> enclosingInterval)
            {
                // Children should already be ordered.
                var children = node.Children;
                int intervalStart = node.Interval.First;
                int intervalEnd = node.Interval.Second;

                int incomingExtreme = intervalStart;
                int outgoingExtreme = intervalEnd;

                var endOfLast = node.Interval.First;

                foreach (var child in children)
                {
                    int startOfCurrent = child.Interval.First;
                    if (endOfLast != startOfCurrent)
                    {
                        var inbetweenNodes = this.Nodes.Skip(startOfCurrent).Take(endOfLast - startOfCurrent + 1);
                        int inExtrem = inbetweenNodes.Where(nd => nd.Index != intervalStart && nd.IncomingNodes.Count > 0).Max(nd => nd.IncomingNodes.Max(e => e.Index));
                        int outExtrem = inbetweenNodes.Where(nd => nd.Index != intervalEnd && nd.OutgoingNodes.Count > 0).Min(nd => nd.OutgoingNodes.Min(e => e.Index));
                        if (inExtrem > incomingExtreme)
                        {
                            incomingExtreme = inExtrem;
                        }

                        if (outExtrem < outgoingExtreme)
                        {
                            outgoingExtreme = outExtrem;
                        }
                    }

                    Pair<int, int> childEnclosingInterval;
                    bool success = this.TryCollapsingIntervalNode(child, out childEnclosingInterval);
                    if (!success)
                    {
                        if (childEnclosingInterval.First > incomingExtreme)
                        {
                            incomingExtreme = childEnclosingInterval.First;
                        }

                        if (childEnclosingInterval.Second < outgoingExtreme)
                        {
                            outgoingExtreme = childEnclosingInterval.Second;
                        }
                    }

                    endOfLast = child.Interval.Second;
                }

                if (endOfLast != intervalEnd)
                {
                    var inbetweenNodes = this.Nodes.Skip(intervalEnd).Take(endOfLast - intervalEnd + 1);
                    int inExtrem = inbetweenNodes.Where(nd => nd.Index != intervalStart && nd.IncomingNodes.Count > 0).Max(nd => nd.IncomingNodes.Max(e => e.Index));
                    int outExtrem = inbetweenNodes.Where(nd => nd.Index != intervalEnd && nd.OutgoingNodes.Count > 0).Min(nd => nd.OutgoingNodes.Min(e => e.Index));
                    if (inExtrem > incomingExtreme)
                    {
                        incomingExtreme = inExtrem;
                    }

                    if (outExtrem < outgoingExtreme)
                    {
                        outgoingExtreme = outExtrem;
                    }
                }

                enclosingInterval = Pair.Create(incomingExtreme, outgoingExtreme);

                if (incomingExtreme > intervalStart || outgoingExtreme < intervalEnd)
                {
                    return false;
                }
                else
                {
                    var collapsedRegex = this.BuildRegexBetweenNodeIndices(intervalStart, intervalEnd, false);
 
                    // Remove any incoming or outgoing from/to internal nodes.
                    for (int i = intervalStart - 1; i > intervalEnd; i--)
                    {
                        var internalNode = this.Nodes[i];
                        internalNode.IncomingNodes.Clear();
                        internalNode.OutgoingNodes.Clear();
                    }

                    // Replace the edge between the interval start and end 
                    var startNode = this.Nodes[intervalStart];
                    var endNode = this.Nodes[intervalEnd];
                    startNode.OutgoingNodes = new HashSet<Edge<TElement>>(startNode.OutgoingNodes.Where(e => e.Index < intervalEnd));
                    endNode.IncomingNodes = new HashSet<Edge<TElement>>(endNode.IncomingNodes.Where(e => e.Index > intervalStart));
                    var edgeToEnd = new Edge<TElement>(intervalEnd, EdgeTypes.CollapsedInterval);
                    var edgeFromStart = new Edge<TElement>(intervalStart, EdgeTypes.CollapsedInterval);
                    edgeToEnd.Regex = collapsedRegex;
                    edgeFromStart.Regex = collapsedRegex;
                    startNode.OutgoingNodes.Add(edgeToEnd);
                    endNode.IncomingNodes.Add(edgeFromStart);

                    return true;
                }
            }

            /// <summary>
            /// Builds incoming edges from outgoing edges.
            /// </summary>
            private void BuildIncomingEdgesFromOutgoingEdges()
            {
                for (int i = 0; i < this.NodeCount; i++)
                {
                    var node = this.Nodes[i];
                    foreach (var ej in node.OutgoingNodes)
                    {
                        var ei = new Edge<TElement>(i);
                        ei.Regex = ej.Regex;
                        this.Nodes[ej.Index].IncomingNodes.Add(ei);
                    }
                }
            }
            
            /// <summary>
            /// Recursively builds the regex tree between two nodes in the graph.
            /// </summary>
            /// <param name="currentIndex">The index of the current node.</param>
            /// <param name="endIndex">The index of the end node.</param>
            /// <param name="includeNodeRegex">Whether to include the regex at the node.</param>
            /// <returns>The recursively-built regex from the current node.</returns>
            private RegexpTreeNode<TElement> BuildRegexBetweenNodeIndices(int currentIndex, int endIndex, bool includeNodeRegex = true)
            {
                if (currentIndex == endIndex)
                {
                    // We've arrived! Don't include node regex for end index.
                    return RegexpTreeNode<TElement>.Empty();
                }

                if (currentIndex < endIndex)
                {
                    throw new ArgumentException(string.Format("Current index {0} cannot be less than end index {1}", currentIndex, endIndex));
                }

                var currentNode = this.Nodes[currentIndex];
                var regex = RegexpTreeNode<TElement>.Nothing();

                foreach (var edge in currentNode.OutgoingNodes)
                {
                    var edgeRegex = RegexpTreeNode<TElement>.Concat(edge.Regex, this.BuildRegexBetweenNodeIndices(edge.Index, endIndex));
                    regex = RegexpTreeNode<TElement>.Or(regex, edgeRegex);
                }

                if (includeNodeRegex)
                {
                    regex = RegexpTreeNode<TElement>.Concat(currentNode.Regex, regex);
                }

                return regex;
            }

            /// <summary>
            /// Comparer for comparing two nodes in an interval tree.
            /// </summary>
            private class IntervalNodeComparer : IComparer<IntervalTreeNode>
            {
                /// <summary>
                /// Compares two nodes in an interval tree.
                /// </summary>
                /// <param name="first">The first interval node.</param>
                /// <param name="second">The second interval node.</param>
                /// <returns>-1, 1, or 0 depending on the comparison.</returns>
                public int Compare(IntervalTreeNode first, IntervalTreeNode second)
                {
                    var firstInt = first.Interval;
                    var secondInt = second.Interval;
                    if (firstInt.First < secondInt.First)
                    {
                        return 1;
                    }
                    else if (firstInt.First > secondInt.First)
                    {
                        return -1;
                    }
                    else if (firstInt.Second < secondInt.Second)
                    {
                        return 1;
                    }
                    else if (firstInt.Second > secondInt.Second)
                    {
                        return -1;
                    }
                    else
                    {
                        return 0;
                    }
                }
            }

            /// <summary>
            /// Comparer for comparing two intervals that orders by containment.
            /// </summary>
            private class ContainmentIntervalComparer : IComparer<Pair<int, int>>
            {
                /// <summary>
                /// Compares two intervals.
                /// </summary>
                /// <param name="x">The first interval.</param>
                /// <param name="y">The second interval.</param>
                /// <returns>-1, 1, or 0 depending on containment, or, if no containment, by left-hand bound.</returns>
                public int Compare(Pair<int, int> x, Pair<int, int> y)
                {
                    if (x.First < y.First)
                    {
                        if (x.Second >= y.Second)
                        {
                            // y strictly contains x
                            return 1;
                        }
                        else
                        {
                            // No containment - order by left bound
                            return 1;
                        }
                    }
                    else if (x.First > y.First)
                    {
                        if (x.Second <= y.Second)
                        {
                            // x strictly contains y
                            return -1;
                        }
                        else
                        {
                            // No containment - order by left bound
                            return -1;
                        }
                    }
                    else
                    {
                        if (x.Second > y.Second)
                        {
                            // y contains x
                            return 1;
                        }
                        else if (x.Second < y.Second)
                        {
                            // x contains y
                            return -1;
                        }
                        else
                        {
                            return 0;
                        }
                    }
                }
            }

            /// <summary>
            /// A node in an interval tree.
            /// </summary>
            private class IntervalTreeNode
            {
                /// <summary>
                /// A static instance of an interval comparer.
                /// </summary>
                private static IComparer<IntervalTreeNode> intervalComparerInstance = new IntervalNodeComparer();

                /// <summary>
                /// Backing field for <see cref="Children"/>.
                /// </summary>
                private List<IntervalTreeNode> children = new List<IntervalTreeNode>();
                
                /// <summary>
                /// Gets or sets the interval for this node.
                /// </summary>
                public Pair<int, int> Interval
                {
                    get;
                    set;
                }

                /// <summary>
                /// Gets the parent of the current node.
                /// </summary>
                public IntervalTreeNode Parent
                {
                    get;
                    private set;
                }

                /// <summary>
                /// Gets the children of the current interval tree node.
                /// </summary>
                public List<IntervalTreeNode> Children
                {
                    get
                    {
                        return this.children;
                    }
                }
                
                /// <summary>
                /// Recursively sorts and merges the children of the current interval node.
                /// </summary>
                public void SortAndMergeChildren()
                {
                    var children = this.children;
                    if (children.Count == 0)
                    {
                        return;
                    }

                    children.Sort(intervalComparerInstance);
                    int lhs = children.First().Interval.First;
                    int rhs = lhs;
                    var mergedList = new List<IntervalTreeNode>();
                    var newNode = new IntervalTreeNode();
                    foreach (var child in children)
                    {
                        var pr = child.Interval;
                        if (pr.First >= rhs && pr.First <= lhs)
                        {
                            // Merge the interval
                            rhs = Math.Min(pr.Second, rhs);
                            newNode.children.AddRange(child.children);
                        }
                        else
                        {
                            newNode.Interval = Pair.Create(lhs, rhs);
                            mergedList.Add(newNode);
                            newNode = new IntervalTreeNode();
                            newNode.children.AddRange(child.children);
                            lhs = pr.First;
                            rhs = pr.Second;
                        }
                    }

                    newNode.Interval = Pair.Create(lhs, rhs);
                    mergedList.Add(newNode);

                    this.children = mergedList;

                    foreach (var mergedChild in mergedList)
                    {
                        mergedChild.SortAndMergeChildren();
                    }
                }

                /// <summary>
                /// Finds the interval tree node that contains a specified interval.
                /// </summary>
                /// <param name="interval">A pair representing the interval.</param>
                /// <returns>The enclosing node, or null.</returns>
                public IntervalTreeNode FindEnclosingNode(Pair<int, int> interval)
                {
                    if (this.Contains(interval))
                    {
                        foreach (var child in this.Children)
                        {
                            if (child.Contains(interval))
                            {
                                return child.FindEnclosingNode(interval);
                            }
                        }

                        return this;
                    }
                    else
                    {
                        return null;
                    }
                }

                /// <summary>
                /// Tries to add an interval to the interval tree.
                /// </summary>
                /// <param name="interval">The interval.</param>
                /// <returns>True of successful, false otherwise.</returns>
                public bool TryAddInterval(Pair<int, int> interval)
                {
                    var parent = this.FindEnclosingNode(interval);
                    if (parent != null)
                    {
                        var childNode = new IntervalTreeNode()
                        {
                            Interval = interval
                        };

                        childNode.Parent = parent;
                        parent.children.Add(childNode);
                        return true;
                    }
                    else
                    {
                        return false;
                    }
                }

                /// <summary>
                /// String representation of the instance.
                /// </summary>
                /// <returns>A string representation of the instance.</returns>
                public override string ToString()
                {
                    return this.Interval.ToString();
                }

                /// <summary>
                /// Tests whether this interval contains another.
                /// </summary>
                /// <param name="interval">A pair representing the other interval.</param>
                /// <returns>True if this interval contains the specified one, false otherwise.</returns>
                private bool Contains(Pair<int, int> interval)
                {
                    return this.Interval.First >= interval.First && this.Interval.Second <= interval.Second;
                }
            }
        }
    }
}
