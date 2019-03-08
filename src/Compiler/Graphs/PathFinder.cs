// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Collections;

namespace Microsoft.ML.Probabilistic.Compiler.Graphs
{
    /// <summary>
    /// Find all elementary paths of a directed graph
    /// </summary>
    /// <typeparam name="NodeType">The node type</typeparam>
    /// <remarks><para>
    /// The paths are described by firing actions according to the pattern:
    /// BeginPath, AddNode, AddNode, ..., AddNode, EndPath, BeginPath, ..., EndPath.
    /// The node on a path will appear in order of the directed edges between them.
    /// </para><para>
    /// Only paths which cannot be made longer are returned, i.e. sub-paths of an elementary path are not returned.
    /// </para></remarks>
    internal class PathFinder<NodeType>
    {
        private IDirectedGraph<NodeType> graph;
        private IndexedProperty<NodeType, bool> isBlocked;
        private Stack<NodeType> stack = new Stack<NodeType>();
        public event Action<NodeType> AddNode;
        public event Action BeginPath , EndPath;

        public PathFinder(IDirectedGraph<NodeType> graph)
        {
            this.graph = graph;
            CanCreateNodeData<NodeType> data = (CanCreateNodeData<NodeType>) graph;
            isBlocked = data.CreateNodeData<bool>(false);
        }

        /// <summary>
        /// Find all paths starting with the given node
        /// </summary>
        /// <param name="node">Starting node</param>
        public void SearchFrom(NodeType node)
        {
            stack.Push(node);
            isBlocked[node] = true;
            bool canExtend = false;
            foreach (NodeType target in graph.TargetsOf(node))
            {
                if (isBlocked[target]) continue;
                // recursive call
                SearchFrom(target);
                canExtend = true;
            }
            isBlocked[node] = false;
            if (!canExtend && stack.Count > 1)
            {
                // path has multiple nodes and cannot be extended
                OnBeginPath();
                Stack<NodeType> temp = new Stack<NodeType>();
                foreach (NodeType nodeOnStack in stack)
                {
                    temp.Push(nodeOnStack);
                }
                foreach (NodeType nodeOnStack in temp)
                {
                    OnAddNode(nodeOnStack);
                }
                OnEndPath();
            }
            stack.Pop();
        }

        public void OnAddNode(NodeType node)
        {
            AddNode?.Invoke(node);
        }

        public void OnBeginPath()
        {
            BeginPath?.Invoke();
        }

        public void OnEndPath()
        {
            EndPath?.Invoke();
        }
    }

    /// <summary>
    /// Find all elementary paths of a directed graph
    /// </summary>
    /// <typeparam name="NodeType">The node type</typeparam>
    /// <typeparam name="EdgeType">The edge type</typeparam>
    /// <remarks><para>
    /// The paths are described by firing actions according to the pattern:
    /// BeginPath, AddEdge, AddEdge, ..., AddEdge, EndPath, BeginPath, ..., EndPath.
    /// The edges on a path will appear in order of their directions.
    /// </para><para>
    /// Only paths which cannot be made longer are returned, i.e. sub-paths of an elementary path are not returned.
    /// </para></remarks>
    internal class PathFinder<NodeType, EdgeType>
    {
        private Converter<NodeType, IEnumerable<EdgeType>> edgesOutOf;
        private Converter<EdgeType, NodeType> targetOf;
        private IndexedProperty<NodeType, bool> isBlocked;
        private Stack<EdgeType> stack = new Stack<EdgeType>();
        public event Action<EdgeType> AddEdge;
        public event Action BeginPath , EndPath;

        public PathFinder(Converter<NodeType, IEnumerable<EdgeType>> edgesOutOf, Converter<EdgeType, NodeType> targetOf, CanCreateNodeData<NodeType> data)
        {
            this.edgesOutOf = edgesOutOf;
            this.targetOf = targetOf;
            isBlocked = data.CreateNodeData<bool>(false);
        }

        public PathFinder(IDirectedGraph<NodeType, EdgeType> graph)
        {
            CanCreateNodeData<NodeType> data = (CanCreateNodeData<NodeType>) graph;
            isBlocked = data.CreateNodeData<bool>(false);
            targetOf = graph.TargetOf;
            edgesOutOf = graph.EdgesOutOf;
        }

        /// <summary>
        /// Find all paths starting with the given node
        /// </summary>
        /// <param name="node">Starting node</param>
        public void SearchFrom(NodeType node)
        {
            isBlocked[node] = true;
            bool foundPath = false;
            foreach (EdgeType edge in edgesOutOf(node))
            {
                NodeType target = targetOf(edge);
                if (isBlocked[target]) continue;
                // recursive call
                stack.Push(edge);
                SearchFrom(target);
                stack.Pop();
                foundPath = true;
            }
            isBlocked[node] = false;
            if (!foundPath && stack.Count > 0)
            {
                OnBeginPath();
                Stack<EdgeType> temp = new Stack<EdgeType>();
                foreach (EdgeType edgeOnStack in stack)
                {
                    temp.Push(edgeOnStack);
                }
                foreach (EdgeType edgeOnStack in temp)
                {
                    OnAddEdge(edgeOnStack);
                }
                OnEndPath();
            }
        }

        public void OnAddEdge(EdgeType edge)
        {
            AddEdge?.Invoke(edge);
        }

        public void OnBeginPath()
        {
            BeginPath?.Invoke();
        }

        public void OnEndPath()
        {
            EndPath?.Invoke();
        }
    }

    /// <summary>
    /// Find all nodes on any path from a source set to a sink set
    /// </summary>
    /// <typeparam name="NodeType"></typeparam>
    internal class NodeOnPathFinder<NodeType>
    {
        private Converter<NodeType, IEnumerable<NodeType>> successors;
        public IndexedProperty<NodeType, bool> isBlocked;
        private IndexedProperty<NodeType, Set<NodeType>> blockedSources;
        private IndexedProperty<NodeType, bool> onPath;
        private Predicate<NodeType> isSink;

        public NodeOnPathFinder(
            Converter<NodeType, IEnumerable<NodeType>> successors,
            CanCreateNodeData<NodeType> data, 
            IndexedProperty<NodeType, bool> onPath, 
            Predicate<NodeType> isSink)
        {
            this.successors = successors;
            isBlocked = data.CreateNodeData<bool>(false);
            blockedSources = data.CreateNodeData<Set<NodeType>>(null);
            this.onPath = onPath;
            this.isSink = isSink;
        }

        public void Clear()
        {
            isBlocked.Clear();
            blockedSources.Clear();
            onPath.Clear();
        }

        public void SearchFrom(NodeType node)
        {
            if (onPath[node]) return;
            if (isSink(node))
            {
                onPath[node] = true;
            }
            isBlocked[node] = true;
            foreach (NodeType target in successors(node))
            {
                if (isBlocked[target]) continue;
                // recursive call
                SearchFrom(target);
                if (onPath[target])
                {
                    onPath[node] = true;
                }
            }
            if (onPath[node]) Unblock(node);
            else
            {
                // at this point, all targets are blocked
                foreach (NodeType target in successors(node))
                {
                    Set<NodeType> blockedSourcesOfTarget = blockedSources[target];
                    if (blockedSourcesOfTarget == null)
                    {
                        blockedSourcesOfTarget = new Set<NodeType>();
                        blockedSources[target] = blockedSourcesOfTarget;
                    }
                    blockedSourcesOfTarget.Add(node);
                }
            }
        }

        private void Unblock(NodeType node)
        {
            isBlocked[node] = false;
            Set<NodeType> blockedSourcesOfNode = blockedSources[node];
            if (blockedSourcesOfNode != null)
            {
                blockedSources[node] = null;
                foreach (NodeType source in blockedSourcesOfNode)
                {
                    if (isBlocked[source]) Unblock(source);
                }
            }
        }
    }

    /// <summary>
    /// Find all edges on any path from a source set to a sink set
    /// </summary>
    /// <typeparam name="NodeType"></typeparam>
    /// <typeparam name="EdgeType"></typeparam>
    internal class EdgeOnPathFinder<NodeType, EdgeType>
    {
        private Converter<NodeType, IEnumerable<EdgeType>> edgesOutOf;
        private Converter<EdgeType, NodeType> targetOf;
        public IndexedProperty<NodeType, bool> isBlocked;
        private IndexedProperty<NodeType, Set<NodeType>> blockedSources;
        private IndexedProperty<EdgeType, bool> onPath;
        private Predicate<NodeType> isSink;

        public EdgeOnPathFinder(
            Converter<NodeType, IEnumerable<EdgeType>> edgesOutOf,
            Converter<EdgeType, NodeType> targetOf,
            CanCreateNodeData<NodeType> data,
            IndexedProperty<EdgeType, bool> onPath,
            Predicate<NodeType> isSink)
        {
            this.edgesOutOf = edgesOutOf;
            this.targetOf = targetOf;
            isBlocked = data.CreateNodeData<bool>(false);
            blockedSources = data.CreateNodeData<Set<NodeType>>(null);
            this.onPath = onPath;
            this.isSink = isSink;
        }

        public void Clear()
        {
            isBlocked.Clear();
            blockedSources.Clear();
            onPath.Clear();
        }

        public void SearchFrom(NodeType node)
        {
            if (isSink(node))
                return;
            isBlocked[node] = true;
            bool foundPath = false;
            foreach (var edge in this.edgesOutOf(node))
            {
                NodeType target = this.targetOf(edge);
                if (isBlocked[target])
                    continue;
                // recursive call
                SearchFrom(target);
                if (!isBlocked[target])
                {
                    onPath[edge] = true;
                    foundPath = true;
                }
            }
            if (foundPath)
                Unblock(node);
            else
            {
                // at this point, all targets are blocked
                foreach (var edge in this.edgesOutOf(node))
                {
                    NodeType target = this.targetOf(edge);
                    Set<NodeType> blockedSourcesOfTarget = blockedSources[target];
                    if (blockedSourcesOfTarget == null)
                    {
                        blockedSourcesOfTarget = new Set<NodeType>();
                        blockedSources[target] = blockedSourcesOfTarget;
                    }
                    blockedSourcesOfTarget.Add(node);
                }
            }
        }

        private void Unblock(NodeType node)
        {
            isBlocked[node] = false;
            Set<NodeType> blockedSourcesOfNode = blockedSources[node];
            if (blockedSourcesOfNode != null)
            {
                blockedSources[node] = null;
                foreach (NodeType source in blockedSourcesOfNode)
                {
                    if (isBlocked[source])
                        Unblock(source);
                }
            }
        }
    }
}