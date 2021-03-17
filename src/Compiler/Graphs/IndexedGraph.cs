// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;
using System.Linq;

namespace Microsoft.ML.Probabilistic.Compiler.Graphs
{
    internal class IndexedGraph : IMutableDirectedGraph<int, int>, IMultigraph<int, int>,
                                  CanCreateNodeData<int>, CanCreateEdgeData<int>
    {
        protected readonly List<List<int>> inEdges;
        protected readonly List<List<int>> outEdges;
        protected readonly List<Edge<int>> edges;
        public bool IsReadOnly;
        public bool NodeCountIsConstant;

        public IndexedGraph()
        {
            inEdges = new List<List<int>>();
            outEdges = new List<List<int>>();
            edges = new List<Edge<int>>();
        }

        public IndexedGraph(int nodeCount)
        {
            inEdges = new List<List<int>>(nodeCount);
            outEdges = new List<List<int>>(nodeCount);
            for (int i = 0; i < nodeCount; i++)
            {
                AddNode();
            }
            edges = new List<Edge<int>>();
            NodeCountIsConstant = true;
        }

        public int SourceOf(int edge)
        {
            return edges[edge].Source;
        }

        public int TargetOf(int edge)
        {
            return edges[edge].Target;
        }

        public IEnumerable<int> EdgesOutOf(int source)
        {
            return outEdges[source];
        }

        public IEnumerable<int> EdgesInto(int target)
        {
            return inEdges[target];
        }

        public IEnumerable<int> Edges
        {
            get { return new Range(0, edges.Count); }
        }

        public int GetEdge(int source, int target)
        {
            if (TryGetEdge(source, target, out int edge)) return edge;
            else throw new EdgeNotFoundException(source, target);
        }

        public bool TryGetEdge(int source, int target, out int edge)
        {
            edge = 0;
            bool found = false;
            foreach (int outEdge in EdgesOutOf(source))
            {
                if (edges[outEdge].Target == target)
                {
                    if (found) throw new AmbiguousEdgeException(source, target);
                    edge = outEdge;
                    found = true;
                }
            }
            return found;
        }

        public int EdgeCount(int source, int target)
        {
            return EdgesOutOf(source).Count(HasTarget(target));
            //return Enumerable.Count(EdgesOutOf(source), HasTarget(target));
        }

        public IEnumerable<int> EdgesLinking(int source, int target)
        {
            foreach (int edge in EdgesOutOf(source))
            {
                if (edges[edge].Target == target) yield return edge;
            }
        }

        /// <summary>
        /// Get an edge handle.
        /// </summary>
        /// <param name="source">A node handle.</param>
        /// <param name="target">A node handle.</param>
        /// <returns>An edge handle if an edge exists.</returns>
        /// <exception cref="EdgeNotFoundException">If there is no edge from source to target.</exception>
        public int GetAnyEdge(int source, int target)
        {
            if (AnyEdge(source, target, out int result))
            {
                return result;
            }
            else
            {
                throw new EdgeNotFoundException(source, target);
            }
        }

        public bool AnyEdge(int source, int target, out int edge)
        {
            edge = 0;
            foreach (int tryEdge in EdgesOutOf(source))
            {
                if (edges[tryEdge].Target == target)
                {
                    edge = tryEdge;
                    return true;
                }
            }
            return false;
        }

        public IEnumerable<int> EdgesOf(int node)
        {
            List<int> result = new List<int>(EdgesInto(node));
            result.AddRange(EdgesOutOf(node));
            return result;
            // This version does not deal properly with self-loops (returns same edge twice)
            //return Enumerable.Join(EdgesInto(node), EdgesOutOf(node));
        }

        public ICollection<int> Nodes
        {
            get { return new Range(0, inEdges.Count); }
        }

        public int EdgeCount()
        {
            return edges.Count;
        }

        public int NeighborCount(int node)
        {
            return NeighborsOf(node).Count();
        }

        public IEnumerable<int> NeighborsOf(int node)
        {
            return SourcesOf(node).Concat(TargetsOf(node));
        }

        public bool ContainsEdge(int source, int target)
        {
            //return Enumerable.Exists(EdgesOutOf(source), HasTarget(target));
            // same as above, but inlined
            foreach (int edge in outEdges[source])
            {
                if (edges[edge].Target == target) return true;
            }
            return false;
        }

        public Func<int, bool> HasTarget(int target)
        {
            return delegate(int edge) { return (edges[edge].Target == target); };
        }

        public int TargetCount(int source)
        {
            return TargetsOf(source).Count();
        }

        public int SourceCount(int target)
        {
            return SourcesOf(target).Count();
        }

        public IEnumerable<int> TargetsOf(int source)
        {
            foreach (int edge in EdgesOutOf(source))
            {
                yield return edges[edge].Target;
            }
        }

        public IEnumerable<int> SourcesOf(int target)
        {
            foreach (int edge in EdgesInto(target))
            {
                yield return edges[edge].Source;
            }
        }

        public int AddEdge(int source, int target)
        {
            if (IsReadOnly) throw new NotSupportedException("Graph is read only");
            int edge = edges.Count;
            edges.Add(new Edge<int>(source, target));
            if (outEdges[source] == null) outEdges[source] = new List<int>();
            outEdges[source].Add(edge);
            if (inEdges[target] == null) inEdges[target] = new List<int>();
            inEdges[target].Add(edge);
            return edge;
        }

        public bool RemoveEdge(int edge)
        {
            if (IsReadOnly) throw new NotSupportedException("Graph is read only");
            throw new Exception("The method or operation is not implemented.");
        }

        public int AddNode()
        {
            if (IsReadOnly) throw new NotSupportedException("Graph is read only");
            if (NodeCountIsConstant) throw new NotSupportedException("The graph size cannot be changed.");
            int node = inEdges.Count;
            inEdges.Add(new List<int>());
            outEdges.Add(new List<int>());
            return node;
        }

        public bool RemoveNodeAndEdges(int node)
        {
            if (IsReadOnly) throw new NotSupportedException("Graph is read only");
            if (NodeCountIsConstant) throw new NotSupportedException("The graph size cannot be changed.");
            throw new Exception("The method or operation is not implemented.");
        }

        public void Clear()
        {
            if (IsReadOnly) throw new NotSupportedException("Graph is read only");
            if (NodeCountIsConstant) throw new NotSupportedException("The graph size cannot be changed.");
            throw new Exception("The method or operation is not implemented.");
        }

        void IMutableGraph<int>.AddEdge(int source, int target)
        {
            if (IsReadOnly) throw new NotSupportedException("Graph is read only");
            AddEdge(source, target);
        }

        public bool RemoveEdge(int source, int target)
        {
            if (IsReadOnly) throw new NotSupportedException("Graph is read only");
            throw new Exception("The method or operation is not implemented.");
        }

        public void ClearEdges()
        {
            if (IsReadOnly) throw new NotSupportedException("Graph is read only");
            throw new Exception("The method or operation is not implemented.");
        }

        public void ClearEdgesOf(int node)
        {
            if (IsReadOnly) throw new NotSupportedException("Graph is read only");
            throw new Exception("The method or operation is not implemented.");
        }

        public void ClearEdgesOutOf(int source)
        {
            if (IsReadOnly) throw new NotSupportedException("Graph is read only");
            throw new Exception("The method or operation is not implemented.");
        }

        public void ClearEdgesInto(int target)
        {
            if (IsReadOnly) throw new NotSupportedException("Graph is read only");
            throw new Exception("The method or operation is not implemented.");
        }

        public IndexedProperty<int, T> CreateNodeData<T>(T defaultValue = default(T))
        {
            if (NodeCountIsConstant)
            {
                T[] data = new T[inEdges.Count];
                IndexedProperty<int, T> prop = MakeIndexedProperty.FromArray<T>(data, defaultValue);
                prop.Clear();
                return prop;
            }
            else
            {
                return new IndexedProperty<int, T>(new Dictionary<int, T>(), defaultValue);
            }
        }

        public IndexedProperty<int, T> CreateEdgeData<T>(T defaultValue = default(T))
        {
            if (IsReadOnly)
            {
                T[] data = new T[edges.Count];
                IndexedProperty<int, T> prop = MakeIndexedProperty.FromArray<T>(data, defaultValue);
                prop.Clear();
                return prop;
            }
            else
            {
                return new IndexedProperty<int, T>(new Dictionary<int, T>(), defaultValue);
            }
        }

        public override string ToString()
        {
            StringBuilder s = new StringBuilder();
            foreach (int node in Nodes)
            {
                s.AppendFormat("{0} -> ", node);
                bool first = true;
                foreach (int target in TargetsOf(node))
                {
                    if (!first) s.Append(" ");
                    else first = false;
                    s.Append(target.ToString(CultureInfo.InvariantCulture));
                }
                s.AppendLine();
            }
            return s.ToString();
        }
    }
}