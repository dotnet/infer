// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Probabilistic.Collections;

namespace Microsoft.ML.Probabilistic.Compiler.Graphs
{
    /// <summary>
    /// Create a graph from a collection of nodes and corresponding adjacency data.
    /// </summary>
    /// <typeparam name="Node">The type of a node handle.</typeparam>
    /// <typeparam name="NodeInfo">The type of a node data structure.</typeparam>
    /// <remarks><para>
    /// This class creates a directed graph object from a collection of node handles and data objects.
    /// The data object is assumed to hold the adjacency information for the handles, in the form
    /// of a delegate sourcesOfNode(data) which returns a collection of node handles.
    /// Each node is given an integer index.
    /// Using these integers you can attach additional data to the graph via CreateNodeData which returns an array.
    /// </para><para>
    /// In the simplest case, NodeInfo can be the same as Node in which case this class just stores
    /// a mapping from nodes to integers and vice versa.  
    /// (The delegate infoOfNode would simply return its argument.)
    /// More generally, NodeInfo can store cached information about the node.
    /// </para></remarks>
    internal class IndexedGraphWrapper<Node, NodeInfo> : IDirectedGraph<int>, CanCreateNodeData<int>
    {
        public NodeInfo[] info;

        /// <summary>Provides an index (into info[]) for each node.</summary>
        public IndexedProperty<Node, int> indexOfNode;

        protected Converter<NodeInfo, ICollection<Node>> sourcesOfNode;
        public ICollection<int>[] targetsOfNode;

        public IndexedGraphWrapper(ICollection<Node> nodes, Converter<Node, NodeInfo> infoOfNode,
                                   Converter<NodeInfo, ICollection<Node>> sourcesOfNode)
            : this(nodes, infoOfNode, sourcesOfNode,
                   new IndexedProperty<Node, int>(new Dictionary<Node, int>()))
        {
        }

        public IndexedGraphWrapper(ICollection<Node> nodes,
                                   Converter<Node, NodeInfo> infoOfNode, Converter<NodeInfo, ICollection<Node>> sourcesOfNode,
                                   IndexedProperty<Node, int> indexOfNode)
        {
            info = new NodeInfo[nodes.Count];
            this.indexOfNode = indexOfNode;
            this.sourcesOfNode = sourcesOfNode;
            //new IndexedProperty<Node,int>(new Dictionary<Node,int>());
            int i = 0;
            foreach (Node node in nodes)
            {
                info[i] = infoOfNode(node);
                indexOfNode[node] = i;
                i++;
            }
            targetsOfNode = new ICollection<int>[info.Length];
        }

        public int TargetCount(int source)
        {
            ICollection<int> targets = targetsOfNode[source];
            return (targets != null) ? targets.Count : 0;
        }

        public int SourceCount(int target)
        {
            return sourcesOfNode(info[target]).Count;
        }

        public IEnumerable<int> TargetsOf(int source)
        {
            ICollection<int> targets = targetsOfNode[source];
            if (targets != null)
            {
                foreach (int target in targets)
                {
                    yield return target;
                }
            }
        }

        public IEnumerable<int> SourcesOf(int target)
        {
            foreach (Node sourceNode in sourcesOfNode(info[target]))
            {
                int source = indexOfNode[sourceNode];
                ICollection<int> targets = targetsOfNode[source];
                if (targets == null) targets = new Set<int>();
                targets.Add(target);
                targetsOfNode[source] = targets;
                yield return source;
            }
        }

        public ICollection<int> Nodes
        {
            get { return new Range(0, info.Length); }
        }

        public int EdgeCount()
        {
            throw new Exception("The method or operation is not implemented.");
        }

        public int NeighborCount(int node)
        {
            return SourceCount(node) + TargetCount(node);
        }

        public IEnumerable<int> NeighborsOf(int node)
        {
            return SourcesOf(node).Concat(TargetsOf(node));
        }

        public bool ContainsEdge(int source, int target)
        {
            foreach (Node targetNode in sourcesOfNode(info[target]))
            {
                if (indexOfNode[targetNode] == target) return true;
            }
            return false;
            //return info[target].Sources.Contains(nodeOf(info[source]));
        }

        public IndexedProperty<int, T> CreateNodeData<T>(T defaultValue)
        {
            T[] data = new T[info.Length];
            IndexedProperty<int, T> prop = MakeIndexedProperty.FromArray<T>(data, defaultValue);
            prop.Clear();
            return prop;
        }
    }
}