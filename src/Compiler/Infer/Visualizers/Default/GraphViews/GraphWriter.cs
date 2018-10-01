// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;

namespace Microsoft.ML.Probabilistic.Compiler.Visualizers.GraphViews
{
    /// <summary>
    /// Class for holding a graph structure with visual attributes
    /// and writing it to the file.
    /// </summary>
    internal abstract class GraphWriter
    {
        /// <summary>
        /// The label of the graph.
        /// </summary>
        public string Name { get; protected set; }

        protected Dictionary<string, Node> nodes;
        protected List<Edge> edges;

        public GraphWriter(string name)
        {
            Name = name;
            nodes = new Dictionary<string, Node>();
            edges = new List<Edge>();
        }

        /// <summary>
        /// Add new node description.
        /// </summary>
        /// <param name="newNode"></param>
        public void AddNode(Node newNode)
        {
            if(nodes.Keys.Contains(newNode.ID))
            {
                if (nodes.Values.Contains(newNode))
                {
                    return;
                }
                else
                {
                    throw new ArgumentException("Another node has the same ID");
                }
            }
            nodes.Add(newNode.ID, newNode);
        }

        /// <summary>
        /// Add new node description with given ID.
        /// </summary>
        /// <param name="id"></param>
        /// <returns>Created empty description for a new node.</returns>
        public Node AddNode(string id)
        {
            if(nodes.Keys.Contains(id))
            {
                throw new ArgumentException("Another node has the same ID");
            }

            Node newNode = new Node(id);
            nodes.Add(id, newNode);
            return newNode;
        }

        /// <summary>
        /// Add two node descriptions and create an edge, that links two given nodes. 
        /// </summary>
        /// <param name="source"></param>
        /// <param name="target"></param>
        /// <returns>Created empty description for a new edge.</returns>
        public Edge AddEdge(Node source, Node target)
        {
            CheckOrAddNode(source);
            CheckOrAddNode(target);

            Edge newEdge = new Edge(source, target);
            edges.Add(newEdge);
            return newEdge;
        }

        protected void CheckOrAddNode(Node node)
        {
            if(nodes.Keys.Contains(node.ID))
            {
                if(nodes.Values.Contains(node))
                {
                    return;
                }
                else
                {
                    throw new ArgumentException("Another node has the same ID");
                }
            }

            nodes.Add(node.ID, node);
        }

        /// <summary>
        /// Add a description of a new edge, that links two nodes with given ID's.
        /// </summary>
        /// <param name="source"></param>
        /// <param name="target"></param>
        /// <returns>Created empty edge description.</returns>
        public Edge AddEdge(string source, string target)
        {
            if(!nodes.Keys.Contains(source) || !nodes.Keys.Contains(target))
            {
                throw new ArgumentException("Graph has no node with one of given ID");
            }

            Edge newEdge = new Edge(nodes[source], nodes[target]);
            edges.Add(newEdge);
            return newEdge;
        }

        /// <summary>
        /// Write the graph description to the file.
        /// </summary>
        /// <param name="path">File name.</param>
        abstract public void Write(string path);
    }



    /// <summary>
    /// Class for holding graph structure with graphic attributes
    /// and writing it to the .dot file.
    /// </summary>
    internal class DotGraphWriter : GraphWriter
    {
        /// <summary>
        /// The legend of the graph.
        /// </summary>
        public Legend Legend { get; set; } = null;

        public DotGraphWriter(string name) : base(name) { }

        /// <summary>
        /// Write the graph description to the file.
        /// </summary>
        /// <param name="path">File name.</param>
        public override void Write(string path)
        {
            using (var writer = new StreamWriter(path, false))
            {
                writer.WriteLine($"digraph {Name}");
                writer.WriteLine("{");

                //default attribute values
                DotWriteHelper.WriteLineIndent(writer, "node [ fontsize=8, style=\"filled, rounded\", fillcolor=transparent ];\n", 1);

                foreach(var n in nodes.Values)
                {
                    DotWriteHelper.Indent(writer, 1);
                    WriteNode(writer, n);
                }

                writer.WriteLine();

                foreach(var e in edges)
                {
                    DotWriteHelper.Indent(writer, 1);
                    WriteEdge(writer, e);
                }

                if (Legend != null)
                {
                    writer.WriteLine();
                    AddLegend(writer);
                }
                writer.WriteLine("}");
            }
        }

        private void WriteNode(StreamWriter writer, Node node)
        {
            writer.Write($"{node.ID} [ ");

            DotWriteHelper.WriteAttribute(writer, "label", null, node.Label);
            DotWriteHelper.WriteAttribute(writer, "fontsize", 0, node.FontSize);
            DotWriteHelper.WriteAttribute(writer, "fontcolor", Color.Empty, node.FontColor, c => ((Color)c).GetHexString());
            DotWriteHelper.WriteAttribute(writer, "fillcolor", Color.Empty, node.FillColor, c => ((Color)c).GetHexString());
            DotWriteHelper.WriteAttribute(writer, "color", Color.Empty, node.BorderColor, c => ((Color)c).GetHexString());
            DotWriteHelper.WriteAttribute(writer, "shape", ShapeStyle.Empty, node.Shape, s => ((ShapeStyle)s).GetStringValue());
            DotWriteHelper.WriteAttribute(writer, "penwidth", 0, node.BorderWidth);

            writer.WriteLine("];");
        }

        private void WriteEdge(StreamWriter writer, Edge edge)
        {
            writer.Write($"{edge.Source.ID} -> {edge.Target.ID} [ ");

            DotWriteHelper.WriteAttribute(writer, "label", null, edge.Label);
            DotWriteHelper.WriteAttribute(writer, "fontsize", 0, edge.FontSize);
            DotWriteHelper.WriteAttribute(writer, "fontcolor", Color.Empty, edge.FontColor, c => ((Color)c).GetHexString());
            DotWriteHelper.WriteAttribute(writer, "color", Color.Empty, edge.Color, c => ((Color)c).GetHexString());
            DotWriteHelper.WriteAttribute(writer, "arrowtail", ArrowheadStyle.Normal, edge.ArrowheadAtSource, s => ((ArrowheadStyle)s).GetStringValue());
            DotWriteHelper.WriteAttribute(writer, "arrowhead", ArrowheadStyle.Normal, edge.ArrowheadAtTarget, s => ((ArrowheadStyle)s).GetStringValue());
            DotWriteHelper.WriteAttribute(writer, "style", EdgeStyle.Normal, edge.Style, s => ((EdgeStyle)s).GetStringValue());
            DotWriteHelper.WriteAttribute(writer, "penwidth", 0, edge.Width);
            DotWriteHelper.WriteAttribute(writer, "dir", false, edge.Reverse, _ => "back");

            writer.WriteLine("];");
        }

        private void AddLegend(StreamWriter writer)
        {
            int count;

            DotWriteHelper.WriteLineIndent(writer, "rankdir=LR;", 1);
            DotWriteHelper.WriteLineIndent(writer, "subgraph legend", 1);
            DotWriteHelper.WriteLineIndent(writer, "{", 1);
            DotWriteHelper.WriteLineIndent(writer, "node [ shape=none ];", 2);

            count = 0;
            DotWriteHelper.WriteLineIndent(writer, "key1", 2);
            DotWriteHelper.WriteLineIndent(writer, "[", 2);
            DotWriteHelper.WriteLineIndent(writer, "label=<", 3);
            DotWriteHelper.WriteLineIndent(writer, "<table border = \"0\" cellpadding = \"2\" cellspacing = \"0\" cellborder = \"0\">", 4);
            foreach (var item in Legend.Items)
            {
                DotWriteHelper.WriteLineIndent(writer, $"<tr><td align=\"right\" port=\"tmp_{count}\">{item.Label}</td></tr>", 5);
                count++;
            }
            DotWriteHelper.WriteLineIndent(writer, "</table>", 4);
            DotWriteHelper.WriteLineIndent(writer, ">", 3);
            DotWriteHelper.WriteLineIndent(writer, "]", 2);

            count = 0;
            DotWriteHelper.WriteLineIndent(writer, "key2", 2);
            DotWriteHelper.WriteLineIndent(writer, "[", 2);
            DotWriteHelper.WriteLineIndent(writer, "label=<", 3);
            DotWriteHelper.WriteLineIndent(writer, "<table border = \"0\" cellpadding = \"2\" cellspacing = \"0\" cellborder = \"0\">", 4);
            foreach (var item in Legend.Items)
            {
                DotWriteHelper.WriteLineIndent(writer, $"<tr><td port=\"tmp_{count}\">&nbsp;</td></tr>", 5);
                count++;
            }
            DotWriteHelper.WriteLineIndent(writer, "</table>", 4);
            DotWriteHelper.WriteLineIndent(writer, ">", 3);
            DotWriteHelper.WriteLineIndent(writer, "]", 2);

            count = 0;
            foreach(var item in Legend.Items)
            {
                DotWriteHelper.WriteIndent(writer, $"key1: tmp_{count}: e->key2: tmp_{count}: w [ ", 2);

                DotWriteHelper.WriteAttribute(writer, "penwidth", 0, item.LineWidth);
                DotWriteHelper.WriteAttribute(writer, "color", Color.Empty, item.LineColor, c => ((Color)c).GetHexString());
                DotWriteHelper.WriteAttribute(writer, "style", EdgeStyle.Normal, item.Style, s => ((EdgeStyle)s).GetStringValue());

                writer.WriteLine("]");
                count++;
            }

            DotWriteHelper.WriteLineIndent(writer, "}", 1);
        }
    }
}
