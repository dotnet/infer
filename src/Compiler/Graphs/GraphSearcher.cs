// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.Graphs
{
    /// <summary>
    /// Labels for depth-first search.
    /// </summary>
    internal enum VisitState
    {
        Unvisited,
        Discovered,
        Visiting,
        Finished
    };

    /// <summary>
    /// Performs depth-first search or breadth-first search on a graph.
    /// </summary>
    /// <typeparam name="NodeType"></typeparam>
    /// <typeparam name="EdgeType"></typeparam>
    internal abstract class GraphSearcher<NodeType, EdgeType>
    {
        public event Action<NodeType> DiscoverNode , FinishNode;
        public event Action<EdgeType> DiscoverEdge , TreeEdge , BackEdge , CrossEdge , FinishTreeEdge;
        public IndexedProperty<NodeType, VisitState> IsVisited;
        protected bool stopped;

        public void Stop()
        {
            stopped = true;
        }

        protected void CreateNodeData(IGraph<NodeType> graph)
        {
            if (graph is CanCreateNodeData<NodeType>)
            {
                IsVisited = ((CanCreateNodeData<NodeType>) graph).CreateNodeData<VisitState>(VisitState.Unvisited);
            }
            else
            {
                IsVisited = new IndexedProperty<NodeType, VisitState>(new Dictionary<NodeType, VisitState>(), VisitState.Unvisited);
            }
        }

        public virtual void Clear()
        {
            IsVisited.Clear();
        }

        public void ClearActions()
        {
            DiscoverNode = null;
            FinishNode = null;
            DiscoverEdge = null;
            TreeEdge = null;
            BackEdge = null;
            CrossEdge = null;
            FinishTreeEdge = null;
        }

        public abstract void SearchFrom(NodeType start);
        public abstract void SearchFrom(IEnumerable<NodeType> startNodes);

        protected void OnDiscoverNode(NodeType node)
        {
            DiscoverNode?.Invoke(node);
        }

        protected void OnFinishNode(NodeType node)
        {
            FinishNode?.Invoke(node);
        }

        protected void OnTreeEdge(EdgeType edge)
        {
            TreeEdge?.Invoke(edge);
        }

        protected void OnBackEdge(EdgeType edge)
        {
            BackEdge?.Invoke(edge);
        }

        protected void OnCrossEdge(EdgeType edge)
        {
            CrossEdge?.Invoke(edge);
        }

        protected void OnDiscoverEdge(EdgeType edge)
        {
            DiscoverEdge?.Invoke(edge);
        }

        protected void OnFinishTreeEdge(EdgeType edge)
        {
            FinishTreeEdge?.Invoke(edge);
        }
    }

    internal abstract class GraphSearcher<NodeType> : GraphSearcher<NodeType, Edge<NodeType>>
    {
        protected Converter<NodeType, IEnumerable<NodeType>> Successors;

        protected GraphSearcher(Converter<NodeType, IEnumerable<NodeType>> successors,
                                IndexedProperty<NodeType, VisitState> isVisited)
        {
            this.Successors = successors;
            this.IsVisited = isVisited;
        }

        protected GraphSearcher(Converter<NodeType, IEnumerable<NodeType>> successors,
                                CanCreateNodeData<NodeType> data)
        {
            this.Successors = successors;
            this.IsVisited = data.CreateNodeData<VisitState>(VisitState.Unvisited);
        }

        protected GraphSearcher(IGraph<NodeType> graph)
        {
            Successors = graph.NeighborsOf;
            CreateNodeData(graph);
        }

        protected GraphSearcher(IDirectedGraph<NodeType> graph)
        {
            Successors = graph.TargetsOf;
            CreateNodeData(graph);
        }
    }
}