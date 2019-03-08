// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Reference:
// [1] "Introduction to Algorithms" by Cormen, Leiserson, and Rivest (1994)
// [2] "An Improved Algorithm for Finding the Strongly Connected Components of a Directed Graph"
// David J. Pearce, Technical Report, 2005
// http://www.mcs.vuw.ac.nz/~djp/files/P05.ps

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Microsoft.ML.Probabilistic.Compiler.Graphs
{
    /// <summary>
    /// Find the strongly connected components of a directed graph.
    /// </summary>
    /// <typeparam name="NodeType">The node type.</typeparam>
    /// <remarks><para>
    /// A strongly connected component is a maximal set of nodes that can all reach each other by a directed path.
    /// Every directed graph has a unique partition of nodes into strongly connected components.
    /// The components form a DAG, since there cannot be any directed cycle among the components.
    /// </para><para>
    /// Given a graph and a set of start nodes, this class enumerates the strongly connected components 
    /// that are reachable from the start nodes.  
    /// The components are described by firing actions according to the pattern:
    /// BeginComponent, AddNode, AddNode, ..., AddNode, EndComponent, BeginComponent, ..., EndComponent.
    /// The nodes in a component will appear in an arbitrary order.
    /// The components will appear in topological order, i.e. there are no edges from
    /// a later component to an earlier component.
    /// </para><para>
    /// The implementation uses depth first search in each direction (Kosaraju's algorithm), as described by Cormen, Leiserson, and Rivest.
    /// </para></remarks>
    internal class StrongComponents<NodeType>
    {
        protected DepthFirstSearch<NodeType> dfsBackward, dfsForward;
        protected Stack<NodeType> finished;
        public event Action<NodeType> AddNode;
        public event Action BeginComponent , EndComponent;

        public StrongComponents(IDirectedGraph<NodeType> graph)
            : this(graph.TargetsOf, graph.SourcesOf, (CanCreateNodeData<NodeType>) graph)
        {
        }

        public StrongComponents(Converter<NodeType, IEnumerable<NodeType>> successors,
                                Converter<NodeType, IEnumerable<NodeType>> predecessors, CanCreateNodeData<NodeType> data)
        {
            dfsForward = new DepthFirstSearch<NodeType>(successors, data);
            finished = new Stack<NodeType>();
            dfsForward.FinishNode += node => finished.Push(node);
            dfsBackward =
                new DepthFirstSearch<NodeType>(node => predecessors(node).Where(source => (dfsForward.IsVisited[source] != VisitState.Unvisited)), data);
            dfsBackward.DiscoverNode += OnAddNode;
        }

        public void SearchFrom(IEnumerable<NodeType> starts)
        {
            dfsForward.SearchFrom(starts);
            ProcessFinished();
        }

        public void SearchFrom(NodeType start)
        {
            dfsForward.SearchFrom(start);
            ProcessFinished();
        }

        protected void ProcessFinished()
        {
            while (finished.Count > 0)
            {
                NodeType start = finished.Pop();
                if (dfsBackward.IsVisited[start] == VisitState.Unvisited)
                {
                    OnBeginComponent();
                    dfsBackward.SearchFrom(start);
                    OnEndComponent();
                }
            }
        }

        public void OnAddNode(NodeType node)
        {
            AddNode?.Invoke(node);
        }

        public void OnBeginComponent()
        {
            BeginComponent?.Invoke();
        }

        public void OnEndComponent()
        {
            EndComponent?.Invoke();
        }
    }

    /// <summary>
    /// Find the strongly connected components of a directed graph.
    /// </summary>
    /// <typeparam name="NodeType">The node type.</typeparam>
    /// <remarks><para>
    /// A strongly connected component is a maximal set of nodes that can all reach each other by a directed path.
    /// Every directed graph has a unique partition of nodes into strongly connected components.
    /// The components form a DAG, since there cannot be any directed cycle among the components.
    /// </para><para>
    /// Given a graph and a set of start nodes, this class enumerates the strongly connected components 
    /// that are reachable from the start nodes.  
    /// The components are described by firing actions according to the pattern:
    /// BeginComponent, AddNode, AddNode, ..., AddNode, EndComponent, BeginComponent, ..., EndComponent.
    /// The nodes in a component will appear in an arbitrary order.
    /// The components will appear in reverse topological order, i.e. there are no edges from
    /// an earlier component to a later component.
    /// </para><para>
    /// The implementation uses Pierce's algorithm (a modification of Tarjan's algorithm):
    /// "An Improved Algorithm for Finding the Strongly Connected Components of a Directed Graph"
    /// David J. Pearce, Technical Report, 2005
    /// http://www.mcs.vuw.ac.nz/~djp/files/P05.ps
    /// </para></remarks>
    internal class StrongComponents2<NodeType>
    {
        private DepthFirstSearch<NodeType> dfs;
        public IndexedProperty<NodeType, int> DiscoverTime, RootDiscoverTime;
        protected Stack<NodeType> finished;
        public event Action<NodeType> AddNode;
        public event Action BeginComponent , EndComponent;
        public int time;

        public StrongComponents2(IDirectedGraph<NodeType> graph)
            : this(graph.TargetsOf, (CanCreateNodeData<NodeType>) graph)
        {
        }

        public void Clear()
        {
            time = 0;
            dfs.Clear();
        }

        public StrongComponents2(Converter<NodeType, IEnumerable<NodeType>> successors, CanCreateNodeData<NodeType> data)
        {
            dfs = new DepthFirstSearch<NodeType>(successors, data);
            finished = new Stack<NodeType>();
            DiscoverTime = data.CreateNodeData<int>(0);
            RootDiscoverTime = data.CreateNodeData<int>(0);
            dfs.DiscoverNode += delegate(NodeType node)
                {
                    DiscoverTime[node] = time;
                    RootDiscoverTime[node] = time;
                    time++;
                };
            dfs.BackEdge += delegate(Edge<NodeType> edge)
                {
                    if (RootDiscoverTime[edge.Target] < RootDiscoverTime[edge.Source])
                        RootDiscoverTime[edge.Source] = RootDiscoverTime[edge.Target];
                };
            dfs.CrossEdge += delegate(Edge<NodeType> edge)
                {
                    if (RootDiscoverTime[edge.Target] < RootDiscoverTime[edge.Source])
                        RootDiscoverTime[edge.Source] = RootDiscoverTime[edge.Target];
                };
            dfs.FinishTreeEdge += delegate(Edge<NodeType> edge)
                {
                    if (RootDiscoverTime[edge.Target] < RootDiscoverTime[edge.Source])
                        RootDiscoverTime[edge.Source] = RootDiscoverTime[edge.Target];
                };
            dfs.FinishNode += delegate(NodeType node)
                {
                    int thisRootDiscoverTime = RootDiscoverTime[node];
                    if (thisRootDiscoverTime < DiscoverTime[node])
                    {
                        // not a root
                        finished.Push(node);
                    }
                    else
                    {
                        // root of a component
                        OnBeginComponent();
                        OnAddNode(node);
                        while (finished.Count > 0 && RootDiscoverTime[finished.Peek()] >= thisRootDiscoverTime)
                        {
                            NodeType child = finished.Pop();
                            OnAddNode(child);
                            RootDiscoverTime[child] = Int32.MaxValue; // prevent child from affecting any other components
                        }
                        RootDiscoverTime[node] = Int32.MaxValue; // prevent node from affecting any other components
                        OnEndComponent();
                    }
                };
        }

        public void SearchFrom(IEnumerable<NodeType> starts)
        {
            dfs.SearchFrom(starts);
        }

        public void SearchFrom(NodeType start)
        {
            dfs.SearchFrom(start);
        }

        public void OnAddNode(NodeType node)
        {
            AddNode?.Invoke(node);
        }

        public void OnBeginComponent()
        {
            BeginComponent?.Invoke();
        }

        public void OnEndComponent()
        {
            EndComponent?.Invoke();
        }
    }

    internal class StrongComponentChecker<NodeType, EdgeType>
    {
        private IDirectedGraph<NodeType, EdgeType> graph;
        private DepthFirstSearch<NodeType, EdgeType> dfs;
        public IndexedProperty<NodeType, int> DiscoverTime, RootDiscoverTime;

        /// <summary>
        /// Modified by SearchFrom
        /// </summary>
        public bool IsStrong;

        public List<EdgeType> RedundantEdges = new List<EdgeType>();
        private int time = 0;

        public StrongComponentChecker(IDirectedGraph<NodeType, EdgeType> graph)
        {
            this.graph = graph;
            dfs = new DepthFirstSearch<NodeType, EdgeType>(graph);
            CanCreateNodeData<NodeType> data = (CanCreateNodeData<NodeType>) graph;
            DiscoverTime = data.CreateNodeData<int>(0);
            RootDiscoverTime = data.CreateNodeData<int>(0);
            dfs.DiscoverNode += delegate(NodeType node)
                {
                    DiscoverTime[node] = time;
                    RootDiscoverTime[node] = time;
                    time++;
                };
            dfs.BackEdge += ProcessEdge2;
            dfs.CrossEdge += ProcessEdge2;
            dfs.FinishTreeEdge += ProcessEdge;
            dfs.FinishNode += delegate(NodeType node)
                {
                    int thisRootDiscoverTime = RootDiscoverTime[node];
                    if (thisRootDiscoverTime < DiscoverTime[node])
                    {
                        // not a root
                    }
                    else
                    {
                        // root of a component
                        if (thisRootDiscoverTime != 0) IsStrong = false;
                    }
                };
        }

        public void ProcessEdge(EdgeType edge)
        {
            NodeType source = graph.SourceOf(edge);
            NodeType target = graph.TargetOf(edge);
            if (RootDiscoverTime[target] < RootDiscoverTime[source])
                RootDiscoverTime[source] = RootDiscoverTime[target];
        }

        public void ProcessEdge2(EdgeType edge)
        {
            NodeType source = graph.SourceOf(edge);
            NodeType target = graph.TargetOf(edge);
            if (RootDiscoverTime[target] < RootDiscoverTime[source])
                RootDiscoverTime[source] = RootDiscoverTime[target];
            else
                RedundantEdges.Add(edge);
        }

        public void SearchFrom(NodeType start)
        {
            IsStrong = true;
            dfs.SearchFrom(start);
            if (time < graph.Nodes.Count) IsStrong = false;
            if (!IsStrong) RedundantEdges.Clear();
        }

        public void Clear()
        {
            IsStrong = false;
            RedundantEdges.Clear();
            time = 0;
            dfs.Clear();
            // do not need to clear DiscoverTime
        }
    }

    /// <summary>
    /// A subgraph with the same nodes but fewer edges.
    /// </summary>
    /// <typeparam name="NodeType"></typeparam>
    /// <typeparam name="EdgeType"></typeparam>
    internal class DirectedGraphFilter<NodeType, EdgeType> : IDirectedGraph<NodeType, EdgeType>, CanCreateNodeData<NodeType>, CanCreateEdgeData<EdgeType>
    {
        private IDirectedGraph<NodeType, EdgeType> graph;
        private Predicate<EdgeType> predicate;

        public DirectedGraphFilter(IDirectedGraph<NodeType, EdgeType> graph, Predicate<EdgeType> predicate)
        {
            this.graph = graph;
            this.predicate = predicate;
        }

        public NodeType SourceOf(EdgeType edge)
        {
            return graph.SourceOf(edge);
        }

        public NodeType TargetOf(EdgeType edge)
        {
            return graph.TargetOf(edge);
        }

        public IEnumerable<EdgeType> EdgesOutOf(NodeType source)
        {
            foreach (EdgeType edge in graph.EdgesOutOf(source))
            {
                if (predicate(edge)) yield return edge;
            }
        }

        public IEnumerable<EdgeType> EdgesInto(NodeType target)
        {
            foreach (EdgeType edge in graph.EdgesInto(target))
            {
                if (predicate(edge)) yield return edge;
            }
        }

        private IEnumerable<EdgeType> AllEdges()
        {
            foreach (EdgeType edge in graph.Edges)
            {
                if (predicate(edge)) yield return edge;
            }
        }

        public IEnumerable<EdgeType> Edges
        {
            get { return AllEdges(); }
        }

        public EdgeType GetEdge(NodeType source, NodeType target)
        {
            EdgeType edge = graph.GetEdge(source, target);
            if (predicate(edge)) return edge;
            else throw new EdgeNotFoundException(source, target);
        }

        public bool TryGetEdge(NodeType source, NodeType target, out EdgeType edge)
        {
            if (graph.TryGetEdge(source, target, out edge)) return predicate(edge);
            else return false;
        }

        public IEnumerable<EdgeType> EdgesOf(NodeType node)
        {
            foreach (EdgeType edge in graph.EdgesOf(node))
            {
                if (predicate(edge)) yield return edge;
            }
        }

        public ICollection<NodeType> Nodes
        {
            get { return graph.Nodes; }
        }

        public int EdgeCount()
        {
            throw new NotImplementedException();
        }

        public int NeighborCount(NodeType node)
        {
            throw new NotImplementedException();
        }

        public IEnumerable<NodeType> NeighborsOf(NodeType node)
        {
            return SourcesOf(node).Concat(TargetsOf(node));
        }

        public bool ContainsEdge(NodeType source, NodeType target)
        {
            EdgeType edge;
            if (graph.TryGetEdge(source, target, out edge)) return predicate(edge);
            else return false;
        }

        public int TargetCount(NodeType source)
        {
            throw new NotImplementedException();
        }

        public int SourceCount(NodeType target)
        {
            throw new NotImplementedException();
        }

        public IEnumerable<NodeType> TargetsOf(NodeType source)
        {
            foreach (EdgeType edge in EdgesOutOf(source)) yield return TargetOf(edge);
        }

        public IEnumerable<NodeType> SourcesOf(NodeType target)
        {
            foreach (EdgeType edge in EdgesInto(target)) yield return SourceOf(edge);
        }

        public IndexedProperty<NodeType, T> CreateNodeData<T>(T defaultValue)
        {
            return ((CanCreateNodeData<NodeType>) graph).CreateNodeData<T>(defaultValue);
        }

        public IndexedProperty<EdgeType, T> CreateEdgeData<T>(T defaultValue)
        {
            return ((CanCreateEdgeData<EdgeType>) graph).CreateEdgeData<T>(defaultValue);
        }

        public override string ToString()
        {
            StringBuilder s = new StringBuilder();
            foreach (NodeType node in Nodes)
            {
                s.AppendFormat("{0} -> ", node);
                bool first = true;
                foreach (NodeType target in TargetsOf(node))
                {
                    if (!first) s.Append(" ");
                    else first = false;
                    s.Append(target.ToString());
                }
                s.AppendLine();
            }
            return s.ToString();
        }
    }
}