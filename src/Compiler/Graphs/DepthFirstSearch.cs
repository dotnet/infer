// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.Graphs
{
    internal class DepthFirstSearch<NodeType> : GraphSearcher<NodeType>
    {
        public DepthFirstSearch(Converter<NodeType, IEnumerable<NodeType>> successors,
                                CanCreateNodeData<NodeType> data)
            : base(successors, data)
        {
            Initialize();
        }

        public DepthFirstSearch(IGraph<NodeType> graph)
            : base(graph)
        {
            Initialize();
        }

        public DepthFirstSearch(IDirectedGraph<NodeType> graph)
            : base(graph)
        {
            Initialize();
        }

        protected struct StackFrame
        {
            public NodeType Node;
            public IEnumerator<NodeType> Successors;

            public StackFrame(NodeType node, IEnumerator<NodeType> successors)
            {
                this.Node = node;
                this.Successors = successors;
            }
        }

        /// <summary>
        /// A stack of nodes for depth-first search.
        /// </summary>
        protected Stack<StackFrame> SearchStack;

        protected void Initialize()
        {
            SearchStack = new Stack<StackFrame>();
        }

        public override void Clear()
        {
            base.Clear();
            SearchStack.Clear();
        }

        public override void SearchFrom(NodeType start)
        {
            if (IsVisited[start] == VisitState.Unvisited)
            {
                Push(start);
                DoSearch();
            }
        }

        public override void SearchFrom(IEnumerable<NodeType> startNodes)
        {
            foreach (NodeType node in startNodes)
            {
                SearchFrom(node);
            }
        }

        public void ForEachStackNode(Action<NodeType> action)
        {
            foreach (StackFrame frame in SearchStack)
            {
                action(frame.Node);
            }
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
                    IsVisited[node] = VisitState.Finished;
                    SearchStack.Pop();
                    OnFinishNode(node);
                    if (SearchStack.Count > 0)
                        OnFinishTreeEdge(new Edge<NodeType>(SearchStack.Peek().Node, node));
                    if (stopped)
                    {
                        SearchStack.Clear();
                        stopped = false;
                    }
                }
            }
        }

        protected void Push(NodeType node)
        {
            SearchStack.Push(new StackFrame(node, Successors(node).GetEnumerator()));
        }

        protected bool PushNextChild(StackFrame frame)
        {
            NodeType node = frame.Node;
            switch (IsVisited[node])
            {
                case VisitState.Unvisited:
                    OnDiscoverNode(node);
                    break;
                case VisitState.Discovered:
                    break;
                case VisitState.Visiting:
                    break;
                case VisitState.Finished:
                    // this happens if we SearchFrom a Finished node.
                    return false;
            }
            // node was previously Unvisited or Discovered
            IsVisited[node] = VisitState.Visiting;
            while (frame.Successors.MoveNext())
            {
                NodeType target = frame.Successors.Current;
                Edge<NodeType> edge = new Edge<NodeType>(node, target);
                OnDiscoverEdge(edge);
                VisitState targetIsVisited = IsVisited[target];
                switch (targetIsVisited)
                {
                    case VisitState.Unvisited:
                        IsVisited[target] = VisitState.Discovered;
                        Push(target);
                        OnDiscoverNode(target);
                        // tree edge
                        OnTreeEdge(edge);
                        return true;
                    case VisitState.Visiting:
                        // back edge
                        OnBackEdge(edge);
                        break;
                    case VisitState.Discovered:
                        // cross edge
                        OnCrossEdge(edge);
                        break;
                    case VisitState.Finished:
                        // cross edge
                        OnCrossEdge(edge);
                        break;
                }
                if (stopped) return false;
            }
            return false;
        }
    }

    internal class DepthFirstSearch<NodeType, EdgeType> : GraphSearcher<NodeType, EdgeType>
    {
        protected IDirectedGraph<NodeType, EdgeType> graph;

        public DepthFirstSearch(IDirectedGraph<NodeType, EdgeType> graph)
        {
            this.graph = graph;
            CreateNodeData(graph);
            Initialize();
        }

        protected struct StackFrame
        {
            public NodeType Node;
            public IEnumerator<EdgeType> EdgesOut;
            public EdgeType TreeEdge;

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

        /// <summary>
        /// A stack of nodes for depth-first search.
        /// </summary>
        protected Stack<StackFrame> SearchStack;

        protected void Initialize()
        {
            SearchStack = new Stack<StackFrame>();
        }

        public override void Clear()
        {
            base.Clear();
            SearchStack.Clear();
        }

        public override void SearchFrom(NodeType start)
        {
            if (IsVisited[start] == VisitState.Unvisited)
            {
                Push(start, default(EdgeType));
                DoSearch();
            }
        }

        public override void SearchFrom(IEnumerable<NodeType> startNodes)
        {
            foreach (NodeType node in startNodes)
            {
                SearchFrom(node);
            }
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
                    IsVisited[node] = VisitState.Finished;
                    SearchStack.Pop();
                    OnFinishNode(node);
                    if (SearchStack.Count > 0)
                        OnFinishTreeEdge(frame.TreeEdge);
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
            switch (IsVisited[node])
            {
                case VisitState.Unvisited:
                    OnDiscoverNode(node);
                    break;
                case VisitState.Discovered:
                    break;
                case VisitState.Visiting:
                    break;
                case VisitState.Finished:
                    // this happens if we SearchFrom a Finished node.
                    return false;
            }
            // node was previously Unvisited or Discovered
            IsVisited[node] = VisitState.Visiting;
            while (frame.EdgesOut.MoveNext())
            {
                EdgeType edge = frame.EdgesOut.Current;
                NodeType target = graph.TargetOf(edge);
                OnDiscoverEdge(edge);
                VisitState targetIsVisited = IsVisited[target];
                switch (targetIsVisited)
                {
                    case VisitState.Unvisited:
                        IsVisited[target] = VisitState.Discovered;
                        Push(target, edge);
                        OnDiscoverNode(target);
                        // tree edge
                        OnTreeEdge(edge);
                        return true;
                    case VisitState.Visiting:
                        // back edge
                        OnBackEdge(edge);
                        break;
                    case VisitState.Discovered:
                        // cross edge
                        OnCrossEdge(edge);
                        break;
                    case VisitState.Finished:
                        // cross edge
                        OnCrossEdge(edge);
                        break;
                }
            }
            return false;
        }
    }
}