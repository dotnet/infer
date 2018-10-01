// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Compiler.Graphs
{
    /// <summary>
    /// An edge with stored endpoints.
    /// </summary>
    /// <typeparam name="NodeType">The type of a node handle.</typeparam>
    /// <remarks>This is a commonly-used interface for an edge object which stores its endpoints.
    /// Edge handles are not required to implement it.
    /// </remarks>
    internal interface IEdge<NodeType>
    {
        NodeType Source { get; }
        NodeType Target { get; }
    }

    internal interface IMutableEdge<NodeType> : IEdge<NodeType>
    {
        new NodeType Source { get; set; }
        new NodeType Target { get; set; }
    }

    /// <summary>
    /// A basic edge object.
    /// </summary>
    /// <typeparam name="NodeType">The type of a node handle.</typeparam>
    internal struct Edge<NodeType> : IEdge<NodeType>
    {
        public NodeType Source, Target;

        public Edge(NodeType source, NodeType target)
        {
            this.Source = source;
            this.Target = target;
        }

        public static Edge<NodeType> New(NodeType source, NodeType target)
        {
            return new Edge<NodeType>(source, target);
        }

        public override string ToString()
        {
            return String.Format("({0},{1})", Source, Target);
        }

        public override bool Equals(object obj)
        {
            if (!(obj is Edge<NodeType>))
                return false;
            Edge<NodeType> that = (Edge<NodeType>)obj;
            return Source.Equals(that.Source) && Target.Equals(that.Target);
        }

        public override int GetHashCode()
        {
            return Hash.Combine(Source.GetHashCode(), Target.GetHashCode());
        }

        #region IEdge<NodeType> Members

        NodeType IEdge<NodeType>.Source
        {
            get { return Source; }
        }

        NodeType IEdge<NodeType>.Target
        {
            get { return Target; }
        }

        #endregion
    }
}