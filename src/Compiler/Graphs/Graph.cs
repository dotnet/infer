// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Probabilistic.Compiler.Reflection;
using Microsoft.ML.Probabilistic.Compiler;

namespace Microsoft.ML.Probabilistic.Compiler.Graphs
{
    internal class NodeDataDictionary<NodeType> : CanCreateNodeData<NodeType>
    {
        public IndexedProperty<NodeType, T> CreateNodeData<T>()
        {
            return new IndexedProperty<NodeType, T>(new Dictionary<NodeType, T>());
        }

        public IndexedProperty<NodeType, T> CreateNodeData<T>(T defaultValue)
        {
            return new IndexedProperty<NodeType, T>(new Dictionary<NodeType, T>(), defaultValue);
        }
    }

    /// <summary>
    /// A directed graph of HasTargets objects.
    /// </summary>
    /// <remarks><p>
    /// Abstractly, a directed graph is a collection of node pairs (node1,node2).
    /// Each pair is called an edge.  An edge from a node to itself (a 
    /// self-loop) is allowed.  Duplicate edges are allowed.  Edges to nodes
    /// outside of the graph are allowed.
    /// </p><p>
    /// A node is added via <c>g.Nodes.Add(node)</c> and an edge is added via
    /// <c>g.AddEdge(node1,node2)</c>.
    /// </p><p>
    /// This implementation supports node labels, which can be any object.
    /// A labeled node is added via <c>g.Nodes.WithLabel(label).Add(node)</c>.
    /// </p><p>
    /// The graph is implemented by an adjacency list which can be singly or 
    /// doubly-linked.
    /// The list is distributed among the nodes of the graph, which hold their
    /// child nodes and possibly also their parent nodes (in the 
    /// doubly-linked case).
    /// Nodes which implement the HasSources interface will be doubly-linked,
    /// and other nodes will be singly-linked.
    /// Thus the graph can be part doubly-linked and part singly-linked.
    /// Doubly-linked nodes are more efficient to remove from the graph.
    /// </p></remarks>
    internal class Graph<NodeType> : NodeDataDictionary<NodeType>,
                                     IMutableDirectedGraph<NodeType>, ILabeledGraph<NodeType, object>,
                                     CanCreateNodeData<NodeType>
        where NodeType : HasTargets<NodeType>
    {
        // nodes must be unique across labels
        protected LabeledSet<NodeType, object> nodes;
        public Func<NodeType> NodeFactory;

        public Graph()
        {
            nodes = new LabeledSet<NodeType, object>("");
        }

        public Graph(Func<NodeType> nodeFactory) : this()
        {
            this.NodeFactory = nodeFactory;
        }

        #region ILabeledGraph methods

        public ILabeledCollection<NodeType, object> Nodes
        {
            get { return this.nodes; }
        }

        #endregion

        #region IGraph methods

        ICollection<NodeType> IGraph<NodeType>.Nodes
        {
            get { return this.nodes; }
        }

        public NodeType AddNode()
        {
            NodeType node = NodeFactory();
            nodes.Add(node);
            return node;
        }

        /// <summary>
        /// Add a directed edge from node to target.
        /// </summary>
        /// <param name="source">The source node.</param>
        /// <param name="target">The target node.</param>
        /// <remarks>The two nodes need not be in the graph, and will not be added to the graph.</remarks>
        public void AddEdge(NodeType source, NodeType target)
        {
            source.Targets.Add(target);
            HasSources<NodeType> intoNode = target as HasSources<NodeType>;
            if (intoNode != null) intoNode.Sources.Add(source);
        }

        public bool ContainsEdge(NodeType source, NodeType target)
        {
            return source.Targets.Contains(target);
        }

        /// <summary>
        /// Remove a directed edge from source to target.
        /// </summary>
        /// <param name="source">The source node.</param>
        /// <param name="target">The target node.</param>
        /// <remarks>If there are multiple edges from source to target, only one is removed.</remarks>
        public virtual bool RemoveEdge(NodeType source, NodeType target)
        {
            HasSources<NodeType> intoNode = target as HasSources<NodeType>;
            if (intoNode != null) intoNode.Sources.Remove(source);
            return source.Targets.Remove(target);
        }

#if zero
    // Remove all edges with a given label connected to a node, but do not 
    // remove the node.  This includes all edges which refer to the node
    // from another node.
    // If node.InwardLabels is null, then inward references will not be
    // be cleared.
        public Graph ClearNodeEdges(HasChildNodesLabeled node, object label)
        {
            foreach(Node toNode in node.GetOutwardEdges(label)) {
                HasParentNodesLabeled intoNode = toNode as HasParentNodesLabeled;
                if(intoNode != null)
                    intoNode.RemoveInwardEdge(node,label);
            }
            node.ClearOutwardEdges(label);
            HasParentNodesLabeled inNode = node as HasParentNodesLabeled;
            if(inNode != null) {
                // doubly-linked case
                foreach(Node fromNode in inNode.GetInwardEdges(label)) {
                    fromNode.RemoveOutwardEdge(node,label);
                }
                inNode.ClearInwardEdges(label);
            } else {
                // singly-linked case
                // must search all nodes in the graph
                foreach(Node fromNode in nodes) {
                    fromNode.RemoveOutwardEdge(node,label);
                }
            }
            return this;
        }
#endif

        public virtual void ClearEdgesOutOf(NodeType source)
        {
            foreach (NodeType target in source.Targets)
            {
                HasSources<NodeType> intoNode = target as HasSources<NodeType>;
                if (intoNode != null) intoNode.Sources.Remove(source);
            }
            source.Targets.Clear();
        }

        public virtual void ClearEdgesInto(NodeType target)
        {
            HasSources<NodeType> inNode = target as HasSources<NodeType>;
            if (inNode != null)
            {
                // doubly-linked case
                foreach (NodeType source in inNode.Sources)
                {
                    source.Targets.Remove(target);
                }
                inNode.Sources.Clear();
            }
            else
            {
                // singly-linked case
                // must search all nodes in the graph
                foreach (NodeType source in nodes)
                {
                    while (source.Targets.Contains(target))
                    {
                        source.Targets.Remove(target);
                    }
                }
            }
        }

        /// <summary>
        /// Remove all edges connected to a node.
        /// </summary>
        /// <param name="node"></param>
        /// <remarks>
        /// The node itself is not removed.  In the singly-linked case, the graph is scanned for
        /// all nodes which link to <paramref name="node"/> and these links are cut.
        /// In this case, there may still be links from outside the graph.
        /// </remarks>
        public virtual void ClearEdgesOf(NodeType node)
        {
            ClearEdgesOutOf(node);
            ClearEdgesInto(node);
        }

        /// <summary>
        /// Remove all edges in the graph.
        /// </summary>
        /// <remarks>In the singly-linked case, there may still be links from outside the graph.
        /// </remarks>
        public virtual void ClearEdges()
        {
            foreach (NodeType node in nodes)
            {
                node.Targets.Clear();
                HasSources<NodeType> inNode = node as HasSources<NodeType>;
                if (inNode != null) inNode.Sources.Clear();
            }
        }

        #endregion

        public virtual int EdgeCount()
        {
            int count = 0;
            foreach (NodeType node in nodes)
            {
                count += node.Targets.Count;
            }
            return count;
        }

        public virtual int NeighborCount(NodeType node)
        {
            return TargetCount(node) + SourceCount(node);
        }

        public virtual int TargetCount(NodeType source)
        {
            return source.Targets.Count;
        }

        public virtual int SourceCount(NodeType target)
        {
            HasSources<NodeType> inNode = target as HasSources<NodeType>;
            if (inNode != null)
            {
                return inNode.Sources.Count;
            }
            else
            {
                // singly-linked case
                // must search all nodes in the graph
                int count = 0;
                foreach (NodeType source in nodes)
                {
                    if (source.Targets.Contains(target))
                    {
                        count++;
                    }
                }
                return count;
            }
        }

        public IEnumerable<NodeType> NeighborsOf(NodeType node)
        {
            foreach (NodeType target in TargetsOf(node))
            {
                yield return target;
            }
            foreach (NodeType source in SourcesOf(node))
            {
                yield return source;
            }
        }

        public IEnumerable<NodeType> TargetsOf(NodeType source)
        {
            return source.Targets;
        }

        public IEnumerable<NodeType> SourcesOf(NodeType target)
        {
            HasSources<NodeType> inNode = target as HasSources<NodeType>;
            if (inNode != null)
            {
                return inNode.Sources;
            }
            else
            {
                // singly-linked case
                // must search all nodes in the graph
                return new SourceEnumerator(this, target);
            }
        }

        internal class SourceEnumerator : IEnumerable<NodeType>
        {
            private Graph<NodeType> graph;
            private NodeType target;

            public IEnumerator<NodeType> GetEnumerator()
            {
                foreach (NodeType source in graph.nodes)
                {
                    if (source.Targets.Contains(target))
                    {
                        yield return source;
                    }
                }
            }

            IEnumerator IEnumerable.GetEnumerator()
            {
                return GetEnumerator();
            }

            public SourceEnumerator(Graph<NodeType> graph, NodeType target)
            {
                this.graph = graph;
                this.target = target;
            }
        }

        // This works for graphs because nodes are unique.
        public virtual object LabelOf(NodeType node)
        {
            foreach (object label in nodes.Labels)
            {
                if (Nodes.WithLabel(label).Contains(node)) return label;
            }
            return null;
        }

        public override string ToString()
        {
            StringBuilder s = new StringBuilder();
            foreach (object label in nodes.Labels)
            {
                s.Append(label).AppendLine(":");
                foreach (NodeType node in nodes.WithLabel(label))
                {
                    foreach (NodeType target in node.Targets)
                    {
                        s.AppendLine(String.Format(" {0} -> {1}", node, target));
                    }
                    HasSources<NodeType> intoNode = node as HasSources<NodeType>;
                    if (intoNode != null)
                    {
                        foreach (NodeType source in intoNode.Sources)
                        {
                            s.AppendLine(String.Format(" {0} <- {1}", node, source));
                        }
                    }
                }
            }
            return s.ToString();
        }

        /// <summary>
        /// Add all nodes from another graph.
        /// </summary>
        /// <param name="that"></param>
        /// <remarks>Nodes are added as references, i.e. they are not cloned.  No new edges are created.</remarks>
        public virtual void Add(Graph<NodeType> that)
        {
            foreach (object label in that.Nodes.Labels)
            {
                foreach (NodeType node in that.Nodes.WithLabel(label))
                {
                    Nodes.WithLabel(label).Add(node);
                }
            }
        }

        public virtual void Clear()
        {
            ClearEdges();
            Nodes.Clear();
        }

        public virtual bool RemoveNodeAndEdges(NodeType node)
        {
            ClearEdgesOf(node);
            return Nodes.Remove(node);
        }

        /// <summary>
        /// Provides node and edge data for an existing graph.
        /// </summary>
        /// <typeparam name="NodeDataType"></typeparam>
        /// <typeparam name="EdgeDataType"></typeparam>
        internal class Data<NodeDataType, EdgeDataType>
        {
            public IDictionary<NodeType, NodeDataType> Nodes;
            public IDictionary<Edge<NodeType>, EdgeDataType> Edges;

            public NodeDataType this[NodeType node]
            {
                get { return Nodes[node]; }
                set { Nodes[node] = value; }
            }

            public EdgeDataType this[NodeType source, NodeType target]
            {
                get { return Edges[new Edge<NodeType>(source, target)]; }
                set { Edges[new Edge<NodeType>(source, target)] = value; }
            }

            public Data()
            {
                Nodes = new Dictionary<NodeType, NodeDataType>();
                Edges = new Dictionary<Edge<NodeType>, EdgeDataType>();
            }
        }
    }

    //---------------------------------------------------------------------------------------------------
    //---------------------------------------------------------------------------------------------------

    /// <summary>
    /// 
    /// </summary>
    /// <typeparam name="NodeType"></typeparam>
    /// <typeparam name="EdgeType"></typeparam>
    /// <param name="source"></param>
    /// <param name="target"></param>
    /// <returns></returns>
    internal delegate EdgeType EdgeFactory<NodeType, EdgeType>(NodeType source, NodeType target);

    /// <summary>
    /// A directed graph with explicit node and edge objects.
    /// </summary>
    /// <typeparam name="NodeType"></typeparam>
    /// <typeparam name="EdgeType"></typeparam>
    internal class Graph<NodeType, EdgeType> : Graph<NodeType>,
                                               IMutableDirectedGraph<NodeType, EdgeType>, IMultigraph<NodeType, EdgeType>,
                                               CanCreateEdgeData<EdgeType>
        where NodeType : HasTargets<NodeType>, HasOutEdges<EdgeType>
        where EdgeType : IEdge<NodeType>
    {
        public Graph()
            : base()
        {
        }

        public Graph(EdgeFactory<NodeType, EdgeType> edgeFactory)
            : base()
        {
            this.EdgeFactory = edgeFactory;
        }

        /// <summary>
        /// A delegate to create edge objects.
        /// </summary>
        /// <remarks>
        /// The delegate only creates the edge object; it does not register the edge with the endpoints.
        /// </remarks>
        public EdgeFactory<NodeType, EdgeType> EdgeFactory;

        public new EdgeType AddEdge(NodeType source, NodeType target)
        {
            EdgeType edge = EdgeFactory(source, target);
            AddEdge(edge);
            return edge;
        }

        public void AddEdge(EdgeType edge)
        {
            edge.Source.OutEdges.Add(edge);
            HasInEdges<EdgeType> intoTarget = edge.Target as HasInEdges<EdgeType>;
            if (intoTarget != null) intoTarget.InEdges.Add(edge);
        }

        public NodeType SourceOf(EdgeType edge)
        {
            return edge.Source;
        }

        public NodeType TargetOf(EdgeType edge)
        {
            return edge.Target;
        }

        public IEnumerable<EdgeType> EdgesOutOf(NodeType source)
        {
            return source.OutEdges;
        }

        public IEnumerable<EdgeType> EdgesInto(NodeType target)
        {
            return ((HasInEdges<EdgeType>) target).InEdges;
        }

        public IEnumerable<EdgeType> Edges
        {
            get
            {
                List<EdgeType> list = new List<EdgeType>();
                foreach (NodeType node in Nodes)
                {
                    foreach (EdgeType edge in node.OutEdges)
                    {
                        list.Add(edge);
                    }
                }
                return list;
            }
        }

        public EdgeType GetEdge(NodeType source, NodeType target)
        {
            EdgeType result;
            if (TryGetEdge(source, target, out result))
            {
                return result;
            }
            else
            {
                throw new EdgeNotFoundException(source, target);
            }
        }

        public virtual bool TryGetEdge(NodeType source, NodeType target, out EdgeType edge)
        {
            bool found = false;
            edge = default(EdgeType);
            foreach (EdgeType anEdge in source.OutEdges)
            {
                if (anEdge.Target.Equals(target))
                {
                    if (found) throw new AmbiguousEdgeException(source, target);
                    found = true;
                    edge = anEdge;
                }
            }
            return found;
        }

        /// <summary>
        /// Get an edge handle.
        /// </summary>
        /// <param name="source">A node handle.</param>
        /// <param name="target">A node handle.</param>
        /// <returns>An edge handle if an edge exists.</returns>
        /// <exception cref="EdgeNotFoundException">If there is no edge from source to target.</exception>
        public EdgeType GetAnyEdge(NodeType source, NodeType target)
        {
            EdgeType result;
            if (AnyEdge(source, target, out result))
            {
                return result;
            }
            else
            {
                throw new EdgeNotFoundException(source, target);
            }
        }

        public virtual bool AnyEdge(NodeType source, NodeType target, out EdgeType edge)
        {
            foreach (EdgeType anEdge in source.OutEdges)
            {
                if (anEdge.Target.Equals(target))
                {
                    edge = anEdge;
                    return true;
                }
            }
            edge = default(EdgeType);
            return false;
        }

        public IEnumerable<EdgeType> EdgesOf(NodeType node)
        {
            List<EdgeType> edges = new List<EdgeType>();
            edges.AddRange(node.OutEdges);
            edges.AddRange(((HasInEdges<EdgeType>) node).InEdges);
            return edges;
        }

        public virtual int EdgeCount(NodeType source, NodeType target)
        {
            int count = 0;
            foreach (EdgeType edge in source.OutEdges)
            {
                if (edge.Target.Equals(target))
                {
                    count++;
                }
            }
            return count;
        }

        public IEnumerable<EdgeType> EdgesLinking(NodeType source, NodeType target)
        {
            List<EdgeType> list = new List<EdgeType>();
            foreach (EdgeType edge in source.OutEdges)
            {
                if (edge.Target.Equals(target))
                {
                    list.Add(edge);
                }
            }
            return list;
        }

        public virtual bool RemoveEdge(EdgeType edge)
        {
            bool removed = edge.Source.OutEdges.Remove(edge);
            HasInEdges<EdgeType> target = edge.Target as HasInEdges<EdgeType>;
            if (target != null) removed = removed && target.InEdges.Remove(edge);
            return removed;
        }

        public override bool RemoveEdge(NodeType source, NodeType target)
        {
            EdgeType edge;
            if (AnyEdge(source, target, out edge))
            {
                return RemoveEdge(edge);
            }
            else
            {
                return false;
            }
        }

        public override void ClearEdgesOutOf(NodeType source)
        {
            foreach (EdgeType edge in source.OutEdges)
            {
                HasInEdges<EdgeType> intoNode = edge.Target as HasInEdges<EdgeType>;
                if (intoNode != null) intoNode.InEdges.Remove(edge);
            }
            source.OutEdges.Clear();
        }

        public override void ClearEdgesInto(NodeType target)
        {
            HasInEdges<EdgeType> inNode = target as HasInEdges<EdgeType>;
            if (inNode != null)
            {
                // doubly-linked case
                foreach (EdgeType edge in inNode.InEdges)
                {
                    edge.Source.OutEdges.Remove(edge);
                }
                inNode.InEdges.Clear();
            }
            else
            {
                // singly-linked case
                // must search all nodes in the graph
                foreach (NodeType source in nodes)
                {
                    EdgeType edge;
                    while (AnyEdge(source, target, out edge))
                    {
                        source.OutEdges.Remove(edge);
                    }
                }
            }
        }

        public override void ClearEdges()
        {
            foreach (NodeType node in nodes)
            {
                node.OutEdges.Clear();
                HasInEdges<EdgeType> inNode = node as HasInEdges<EdgeType>;
                if (inNode != null) inNode.InEdges.Clear();
            }
        }

        public override int EdgeCount()
        {
            int count = 0;
            foreach (NodeType node in nodes)
            {
                count += node.OutEdges.Count;
            }
            return count;
        }

        public override int TargetCount(NodeType source)
        {
            return source.OutEdges.Count;
        }

        public override int SourceCount(NodeType target)
        {
            HasInEdges<EdgeType> inNode = target as HasInEdges<EdgeType>;
            if (inNode != null)
            {
                return inNode.InEdges.Count;
            }
            else
            {
                // singly-linked case
                // must search all nodes in the graph
                int count = 0;
                foreach (NodeType source in nodes)
                {
                    if (source.Targets.Contains(target))
                    {
                        count++;
                    }
                }
                return count;
            }
        }

        /// <summary>
        /// Copy edge data to another source and target.
        /// </summary>
        /// <param name="edge"></param>
        /// <param name="source"></param>
        /// <param name="target"></param>
        /// <returns>A new edge with the same data as edge but between source and target.</returns>
        public EdgeType CopyEdge(EdgeType edge, NodeType source, NodeType target)
        {
            if (edge is ICloneable && edge is IMutableEdge<NodeType>)
            {
                EdgeType newEdge = (EdgeType) ((ICloneable) edge).Clone();
                IMutableEdge<NodeType> mutEdge = (IMutableEdge<NodeType>) newEdge;
                mutEdge.Source = source;
                mutEdge.Target = target;
                AddEdge(newEdge);
                return newEdge;
            }
            else
            {
                return AddEdge(source, target);
            }
        }

        /// <summary>
        /// Copy edges from one node to another.
        /// </summary>
        /// <param name="node"></param>
        /// <param name="node2"></param>
        /// <remarks>
        /// For every edge (node,x) or (x,node), an edge (node2,x) or (x,node2) is created, with the
        /// same label.
        /// The existing edges of <paramref name="node2"/> are left unchanged.
        /// </remarks>
        public void CopyEdges(NodeType node, NodeType node2)
        {
            foreach (EdgeType edge in node.OutEdges)
            {
                CopyEdge(edge, node2, edge.Target);
            }
            HasInEdges<EdgeType> inNode = node as HasInEdges<EdgeType>;
            if (inNode != null)
            {
                foreach (EdgeType edge in inNode.InEdges)
                {
                    CopyEdge(edge, edge.Source, node2);
                }
            }
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="g"></param>
        /// <remarks>Clones all nodes in <paramref name="g"/>, preserving edges between them
        /// and to nodes outside the graph.</remarks>
        public Graph(Graph<NodeType, EdgeType> g)
        {
            EdgeFactory = g.EdgeFactory;
            // first clone the nodes
            Dictionary<NodeType, NodeType> newNodes = new Dictionary<NodeType, NodeType>();
            foreach (object label in g.Nodes.Labels)
            {
                foreach (NodeType node in g.Nodes.WithLabel(label))
                {
                    // clone the node contents, but not its neighbors
                    NodeType newNode = (NodeType) Invoker.Clone(node);
                    newNode.OutEdges.Clear();
                    if (newNode is HasInEdges<EdgeType>)
                        ((HasInEdges<EdgeType>) newNode).InEdges.Clear();
                    Nodes.WithLabel(label).Add(newNode);
                    newNodes[node] = newNode;
                }
            }
            // now clone the edges
            foreach (NodeType node in g.Nodes)
            {
                NodeType newNode = newNodes[node];
                foreach (EdgeType edge in node.OutEdges)
                {
                    NodeType target;
                    if (!newNodes.TryGetValue(edge.Target, out target))
                    {
                        // edge to a node outside the graph
                        target = edge.Target;
                    }
                    CopyEdge(edge, newNode, target);
                }
                HasInEdges<EdgeType> inNode = node as HasInEdges<EdgeType>;
                if (inNode != null)
                {
                    foreach (EdgeType edge in inNode.InEdges)
                    {
                        if (newNodes.ContainsKey(edge.Source)) continue;
                        // edge from a node outside the graph
                        CopyEdge(edge, edge.Source, newNode);
                    }
                }
            }
        }

        public virtual object Clone()
        {
            return new Graph<NodeType, EdgeType>(this);
        }

        /// <summary>
        /// Check that parent and child edges match.
        /// </summary>
        [System.Diagnostics.ConditionalAttribute("DEBUG")]
        public void CheckValid()
        {
            foreach (NodeType node in Nodes)
            {
                // check out edges of node
                foreach (EdgeType edge in node.OutEdges)
                {
                    HasInEdges<EdgeType> inTarget = edge.Target as HasInEdges<EdgeType>;
                    if (inTarget == null) continue;
                    if (!inTarget.InEdges.Contains(edge))
                    {
                        throw new InferCompilerException(node + " -> " + edge.Target + " has no backward pointer");
                    }
                }
                // check in edges of node
                HasInEdges<EdgeType> inNode = node as HasInEdges<EdgeType>;
                if (inNode == null) continue;
                foreach (EdgeType edge in inNode.InEdges)
                {
                    if (!edge.Source.OutEdges.Contains(edge))
                    {
                        throw new InferCompilerException(node + " <- " + edge.Source + " has no backward pointer");
                    }
                }
            }
        }

        public override string ToString()
        {
            StringBuilder s = new StringBuilder();
            foreach (object label in nodes.Labels)
            {
                s.Append(label).AppendLine(":");
                foreach (NodeType source in nodes.WithLabel(label))
                {
                    foreach (EdgeType edge in source.OutEdges)
                    {
                        s.Append(" ").AppendLine(edge.ToString());
                    }
#if false
                    HasInEdges<EdgeType> target = source as HasInEdges<EdgeType>;
                    if (target != null) {
                        foreach (EdgeType edge in target.InEdges) {
                            s.Append(" ").AppendLine(edge.ToString());
                        }
                    }
#endif
                }
            }
            return s.ToString();
        }

        public IndexedProperty<EdgeType, T> CreateEdgeData<T>()
        {
            return new IndexedProperty<EdgeType, T>(new Dictionary<EdgeType, T>(), default(T));
        }

        public IndexedProperty<EdgeType, T> CreateEdgeData<T>(T defaultValue)
        {
            return new IndexedProperty<EdgeType, T>(new Dictionary<EdgeType, T>(), defaultValue);
        }

        /// <summary>
        /// Provides node and edge data for an existing graph.
        /// </summary>
        /// <typeparam name="NodeDataType"></typeparam>
        /// <typeparam name="EdgeDataType"></typeparam>
        public new class Data<NodeDataType, EdgeDataType>
        {
            public IDictionary<NodeType, NodeDataType> Nodes;
            public IDictionary<EdgeType, EdgeDataType> Edges;

            public NodeDataType this[NodeType node]
            {
                get { return Nodes[node]; }
                set { Nodes[node] = value; }
            }

            public EdgeDataType this[EdgeType edge]
            {
                get { return Edges[edge]; }
                set { Edges[edge] = value; }
            }

#if false
            public EdgeDataType this[NodeType source, NodeType target, EdgeType label]
            {
                get
                {
                    return Edges[new EdgeIndexType(source, target, label)];
                }
                set
                {
                    Edges[new EdgeIndexType(source, target, label)] = value;
                }
            }
#endif

            public Data()
            {
                Nodes = new Dictionary<NodeType, NodeDataType>();
                Edges = new Dictionary<EdgeType, EdgeDataType>();
            }
        }
    }
}