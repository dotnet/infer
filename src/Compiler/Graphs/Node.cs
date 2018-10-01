// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;

namespace Microsoft.ML.Probabilistic.Compiler.Graphs
{
    internal interface HasTargets<T>
    {
        ICollection<T> Targets { get; }
    }

    internal interface HasSources<T>
    {
        ICollection<T> Sources { get; }
    }

    internal interface HasSourcesAndTargets<T> : HasSources<T>, HasTargets<T>
    {
    }

    /// <summary>
    /// Stores a list of source nodes and target nodes. (Does not store edges.)
    /// </summary>
    /// <typeparam name="T">Node type to link to (usually a DirectedNode itself).</typeparam>
    internal class DirectedNode<T> : HasSourcesAndTargets<T>
    {
        protected List<T> targets, sources;

        public ICollection<T> Targets
        {
            get { return targets; }
        }

        public ICollection<T> Sources
        {
            get { return sources; }
        }

        public ICollection<T> Neighbors
        {
            get { return new JoinCollections<T>(sources, targets); }
        }

        public DirectedNode()
        {
            targets = new List<T>();
            sources = new List<T>();
        }

        public DirectedNode(HasSourcesAndTargets<T> node)
        {
            // clone the collections, but not the referenced nodes
            targets = new List<T>(node.Targets);
            sources = new List<T>(node.Sources);
        }
    }

    internal interface HasOutEdges<E>
    {
        ICollection<E> OutEdges { get; }
    }

    internal interface HasInEdges<E>
    {
        ICollection<E> InEdges { get; }
    }

    internal interface HasInAndOutEdges<E> : HasInEdges<E>, HasOutEdges<E>
    {
    }

    /// <summary>
    /// Stores a list of OutEdges and InEdges.
    /// </summary>
    /// <typeparam name="T">Node type used by the edges.</typeparam>
    /// <typeparam name="E">Edge type to store.</typeparam>
    internal class DirectedNode<T, E> : HasSourcesAndTargets<T>, HasInAndOutEdges<E>
        where E : IEdge<T>
    {
        protected List<E> outEdges, inEdges;

        public ICollection<E> OutEdges
        {
            get { return outEdges; }
        }

        public ICollection<E> InEdges
        {
            get { return inEdges; }
        }

        public ICollection<T> Targets
        {
            get { return outEdges.ConvertAll<T>(delegate(E edge) { return edge.Target; }).AsReadOnly(); }
        }

        public ICollection<T> Sources
        {
            get { return inEdges.ConvertAll<T>(delegate(E edge) { return edge.Source; }).AsReadOnly(); }
        }

        public DirectedNode()
        {
            outEdges = new List<E>();
            inEdges = new List<E>();
        }

        public DirectedNode(HasInAndOutEdges<E> node)
        {
            // clone the collections, but not the referenced edges
            outEdges = new List<E>(node.OutEdges);
            inEdges = new List<E>(node.InEdges);
        }
    }

    /// <summary>
    /// Directed graph node holding data of type T
    /// </summary>
    /// <typeparam name="T">The data type</typeparam>
    internal class BasicNode<T> : DirectedNode<BasicNode<T>>
    {
        public T Data;

        public BasicNode(T data)
            : base()
        {
            Data = data;
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="node"></param>
        public BasicNode(BasicNode<T> node)
            : base(node)
        {
            Data = node.Data;
        }

        public override string ToString()
        {
            if (object.ReferenceEquals(Data, null)) return "(Node)";
            return Data.ToString();
        }
    }

    internal class BasicNode : DirectedNode<BasicNode>
    {
        public object Data;

        public BasicNode(object data)
            : base()
        {
            Data = data;
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="node"></param>
        public BasicNode(BasicNode node)
            : base(node)
        {
            Data = node.Data;
        }

        public override string ToString()
        {
            if (Data == null) return "(Node)";
            return Data.ToString();
        }
    }

    internal class BasicEdgeNode : DirectedNode<BasicEdgeNode, Edge<BasicEdgeNode>>
    {
        public object Data;

        public BasicEdgeNode(object data)
            : base()
        {
            Data = data;
        }

        /// <summary>
        /// Copy constructor.
        /// </summary>
        /// <param name="node"></param>
        public BasicEdgeNode(BasicEdgeNode node)
            : base(node)
        {
            Data = node.Data;
        }

        public override string ToString()
        {
            if (Data == null) return "(Node)";
            return Data.ToString();
        }
    }
}