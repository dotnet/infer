// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Collections;

namespace Microsoft.ML.Probabilistic.Compiler.Graphs
{
    /// <summary>
    /// Find all elementary cycles of a directed graph
    /// </summary>
    /// <typeparam name="NodeType">The node type</typeparam>
    /// <remarks><para>
    /// The cycles are described by firing actions according to the pattern:
    /// BeginCycle, AddNode, AddNode, ..., AddNode, EndCycle, BeginCycle, ..., EndCycle.
    /// The nodes in a cycle will appear in order of the directed edges between them.
    /// </para><para>
    /// The algorithm comes from:
    /// "Finding all the Elementary Circuits of a Directed Graph"
    /// Donald B. Johnson 
    /// SIAM Journal on Computing  (1975) 
    /// http://dutta.csc.ncsu.edu/csc791_spring07/wrap/circuits_johnson.pdf
    /// The runtime is O((n+e)(c+1)) where n is the number of nodes, e is the number of edges, and c
    /// is the number of cycles in the graph.
    /// </para></remarks>
    internal class CycleFinder<NodeType>
    {
        private IDirectedGraph<NodeType> graph;
        private IndexedProperty<NodeType, bool> isBlocked;
        private IndexedProperty<NodeType, Set<NodeType>> blockedSources;
        private Stack<NodeType> stack = new Stack<NodeType>();
        private Set<NodeType> excluded = new Set<NodeType>();
        public event Action<NodeType> AddNode;
        public event Action BeginCycle , EndCycle;

        public CycleFinder(IDirectedGraph<NodeType> graph)
        {
            this.graph = graph;
            CanCreateNodeData<NodeType> data = (CanCreateNodeData<NodeType>) graph;
            isBlocked = data.CreateNodeData<bool>(false);
            blockedSources = data.CreateNodeData<Set<NodeType>>(null);
        }

        public void Search()
        {
            foreach (NodeType node in graph.Nodes)
            {
                SearchFrom(node, node);
                // we have already found all cycles containing this node, so we exclude it from future searches
                excluded.Add(node);
                foreach (NodeType node2 in graph.Nodes)
                {
                    isBlocked[node2] = false;
                    blockedSources[node2] = null;
                }
            }
        }

        /// <summary>
        /// Find cycles containing root 
        /// </summary>
        /// <param name="node"></param>
        /// <param name="root"></param>
        /// <returns></returns>
        private bool SearchFrom(NodeType node, NodeType root)
        {
            bool foundCycle = false;
            stack.Push(node);
            isBlocked[node] = true;
            foreach (NodeType target in graph.TargetsOf(node))
            {
                if (excluded.Contains(target)) continue;
                if (target.Equals(root))
                {
                    foundCycle = true;
                    OnBeginCycle();
                    Stack<NodeType> temp = new Stack<NodeType>();
                    foreach (NodeType nodeOnStack in stack)
                    {
                        temp.Push(nodeOnStack);
                    }
                    foreach (NodeType nodeOnStack in temp)
                    {
                        OnAddNode(nodeOnStack);
                    }
                    OnEndCycle();
                }
                else if (!isBlocked[target])
                {
                    // recursive call
                    if (SearchFrom(target, root)) foundCycle = true;
                }
            }
            // at this point, we could always set isBlocked[node]=false,
            // but as an optimization we leave it set if no cycle was discovered,
            // to prevent repeated searching of the same paths.
            if (foundCycle) Unblock(node);
            else
            {
                // at this point, all targets are blocked
                foreach (NodeType target in graph.TargetsOf(node))
                {
                    if (excluded.Contains(target)) continue;
                    Set<NodeType> blockedSourcesOfTarget = blockedSources[target];
                    if (blockedSourcesOfTarget == null)
                    {
                        blockedSourcesOfTarget = new Set<NodeType>();
                        blockedSources[target] = blockedSourcesOfTarget;
                    }
                    blockedSourcesOfTarget.Add(node);
                }
            }
            stack.Pop();
            return foundCycle;
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

        public void OnAddNode(NodeType node)
        {
            AddNode?.Invoke(node);
        }

        public void OnBeginCycle()
        {
            BeginCycle?.Invoke();
        }

        public void OnEndCycle()
        {
            EndCycle?.Invoke();
        }
    }

    /// <summary>
    /// Find all elementary cycles of a directed graph
    /// </summary>
    /// <typeparam name="NodeType">The node type</typeparam>
    /// <typeparam name="EdgeType">The edge type</typeparam>
    /// <remarks><para>
    /// The cycles are described by firing actions according to the pattern:
    /// BeginCycle, AddEdge, AddEdge, ..., AddEdge, EndCycle, BeginCycle, ..., EndCycle.
    /// The edges in a cycle will appear in order of their directions.
    /// </para><para>
    /// The algorithm comes from:
    /// "Finding all the Elementary Circuits of a Directed Graph"
    /// Donald B. Johnson 
    /// SIAM Journal on Computing  (1975) 
    /// http://dutta.csc.ncsu.edu/csc791_spring07/wrap/circuits_johnson.pdf
    /// The runtime is O((n+e)(c+1)) where n is the number of nodes, e is the number of edges, and c
    /// is the number of cycles in the graph.
    /// </para></remarks>
    internal class CycleFinder<NodeType, EdgeType>
    {
        public event Action<EdgeType> AddEdge;
        public event Action BeginCycle , EndCycle;
        private IDirectedGraph<NodeType, EdgeType> graph;
        private IndexedProperty<NodeType, bool> isBlocked;
        private IndexedProperty<NodeType, Set<NodeType>> blockedSources;
        private Set<NodeType> excluded = new Set<NodeType>();
        protected Stack<StackFrame> SearchStack = new Stack<StackFrame>();
        private NodeType root;

        protected class StackFrame
        {
            public NodeType Node;
            public IEnumerator<EdgeType> EdgesOut;
            public EdgeType TreeEdge;
            public bool foundCycle;

            public StackFrame(NodeType node, IEnumerator<EdgeType> edgesOut, EdgeType treeEdge)
            {
                this.Node = node;
                this.EdgesOut = edgesOut;
                this.TreeEdge = treeEdge;
            }

            public override string ToString()
            {
                return Node.ToString();
            }
        }

        public CycleFinder(IDirectedGraph<NodeType, EdgeType> graph)
        {
            this.graph = graph;
            CanCreateNodeData<NodeType> data = (CanCreateNodeData<NodeType>) graph;
            isBlocked = data.CreateNodeData<bool>(false);
            blockedSources = data.CreateNodeData<Set<NodeType>>(null);
        }

        public void Search()
        {
            foreach (NodeType node in graph.Nodes)
            {
                SearchFrom(node);
                // we have already found all cycles containing this node, so we exclude it from future searches
                excluded.Add(node);
                foreach (NodeType node2 in graph.Nodes)
                {
                    isBlocked[node2] = false;
                    blockedSources[node2] = null;
                }
            }
        }

        /// <summary>
        /// Find cycles containing root 
        /// </summary>
        /// <param name="node"></param>
        /// <returns></returns>
        private void SearchFrom(NodeType node)
        {
            root = node;
            Push(node, default(EdgeType));
            DoSearch();
        }

        protected void DoSearch()
        {
            while (SearchStack.Count > 0)
            {
                StackFrame frame = SearchStack.Peek();
                if (!PushNextChild(frame))
                {
                    // all children have been visited, so we can remove ourselves from the stack.
                    NodeType node = frame.Node;
                    // at this point, we could always set isBlocked[node]=false,
                    // but as an optimization we leave it set if no cycle was discovered,
                    // to prevent repeated searching of the same paths.
                    if (frame.foundCycle) Unblock(node);
                    else
                    {
                        // at this point, all targets are blocked
                        foreach (NodeType target in graph.TargetsOf(node))
                        {
                            if (excluded.Contains(target)) continue;
                            Set<NodeType> blockedSourcesOfTarget = blockedSources[target];
                            if (blockedSourcesOfTarget == null)
                            {
                                blockedSourcesOfTarget = new Set<NodeType>();
                                blockedSources[target] = blockedSourcesOfTarget;
                            }
                            blockedSourcesOfTarget.Add(node);
                        }
                    }
                    SearchStack.Pop();
                    if (frame.foundCycle && SearchStack.Count > 0)
                    {
                        SearchStack.Peek().foundCycle = true;
                    }
                }
            }
        }

        protected void Push(NodeType node, EdgeType treeEdge)
        {
            SearchStack.Push(new StackFrame(node, graph.EdgesOutOf(node).GetEnumerator(), treeEdge));
        }

        protected bool PushNextChild(StackFrame frame)
        {
            NodeType node = frame.Node;
            isBlocked[node] = true;
            while (frame.EdgesOut.MoveNext())
            {
                EdgeType edge = frame.EdgesOut.Current;
                NodeType target = graph.TargetOf(edge);
                if (excluded.Contains(target)) continue;
                if (target.Equals(root))
                {
                    frame.foundCycle = true;
                    OnBeginCycle();
                    Stack<EdgeType> temp = new Stack<EdgeType>();
                    foreach (StackFrame frame2 in SearchStack)
                    {
                        temp.Push(frame2.TreeEdge);
                    }
                    // the last TreeEdge is a dummy
                    temp.Pop();
                    foreach (EdgeType edgeOnStack in temp)
                    {
                        OnAddEdge(edgeOnStack);
                    }
                    OnAddEdge(edge);
                    OnEndCycle();
                }
                else if (!isBlocked[target])
                {
                    // recursive call
                    Push(target, edge);
                    return true;
                }
            }
            return false;
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

        public void OnAddEdge(EdgeType edge)
        {
            AddEdge?.Invoke(edge);
        }

        public void OnBeginCycle()
        {
            BeginCycle?.Invoke();
        }

        public void OnEndCycle()
        {
            EndCycle?.Invoke();
        }
    }
}