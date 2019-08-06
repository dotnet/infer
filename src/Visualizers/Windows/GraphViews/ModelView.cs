// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Windows.Forms;
using System.IO;
using System.Xml;
using System.Linq;
using System.Runtime.CompilerServices;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.Msagl.GraphViewerGdi;
using Microsoft.Msagl.Drawing;

namespace Microsoft.ML.Probabilistic.Compiler.Visualizers
{
    internal class ModelView
    {
        private GViewer gviewer = new GViewer();

        internal ModelBuilder modelBuilder;

        internal ModelView()
        {
        }

        internal ModelView(ModelBuilder model)
        {
            Model = model;
        }

        /// <summary>
        /// The model being shown in the view
        /// </summary>
        internal ModelBuilder Model
        {
            get { return modelBuilder; }
            set
            {
                modelBuilder = value;
                OnModelChange();
            }
        }

        protected void OnModelChange()
        {
            var g = new MsaglWriter("Model");
            GraphViews.GraphBuilder.Add(g, modelBuilder, false);
            gviewer.Graph = g.ToGraph();
        }

        // This must not be inlined to avoid dependence on MSAGL when the method is never called.
        [MethodImpl(MethodImplOptions.NoInlining)]
        public void ShowInForm(string title, bool maximise)
        {
            Form f = WindowsVisualizer.FormHelper.ShowInForm(gviewer, title, maximise);
            Application.Run(f);
        }

        /// <summary>
        /// Write the graph to a file in DGML format
        /// </summary>
        /// <param name="path"></param>
        public void WriteDgml(string path)
        {
            // References:
            // http://en.wikipedia.org/wiki/DGML
            // http://msdn.microsoft.com/library/ee842619.aspx
            // http://ceyhunciper.wordpress.com/category/dgml/
            // http://www.lovettsoftware.com/blogengine.net/post/2010/07/16/DGML-with-Style.aspx
            GraphViews.GraphWriter g = new DgmlWriter("Model");
            GraphViews.GraphBuilder.Add(g, modelBuilder, true);
            g.Write(path);
        }
    }

    internal class DgmlWriter : GraphViews.GraphWriter
    {
        public DgmlWriter(string name) : base(name) { }

        public override void Write(string path)
        {
            var settings = new XmlWriterSettings();
            settings.Indent = true;
            Func<Color, string> colorToString = c => c.ToString().Trim('"');
            using (var writer = XmlWriter.Create(path, settings))
            {
                writer.WriteStartElement("DirectedGraph", "http://schemas.microsoft.com/vs/2009/dgml");
                writer.WriteStartElement("Nodes");
                foreach (string key in nodes.Keys)
                {
                    GraphViews.Node node = (GraphViews.Node)nodes[key];
                    writer.WriteStartElement("Node");
                    writer.WriteAttributeString("Id", key);
                    writer.WriteAttributeString("Label", node.Label);
                    writer.WriteAttributeString("FontSize", node.FontSize.ToString());
                    writer.WriteAttributeString("Foreground", node.FontColor.ToString());
                    writer.WriteAttributeString("Background", node.FillColor.ToString());
                    if (node.Shape == GraphViews.ShapeStyle.None)
                        writer.WriteAttributeString("Shape", "None");
                    if (node.UserData is GraphViews.GraphBuilder.Group)
                        writer.WriteAttributeString("Group", "Expanded");
                    else
                    {
                        // The Shape attribute in DGML can only be used to remove the outline.
                        // To actually change the shape, use the NodeRadius attribute.
                        if (node.Shape == GraphViews.ShapeStyle.Box)
                            writer.WriteAttributeString("NodeRadius", "0");
                        else if (node.Shape == GraphViews.ShapeStyle.Ellipse)
                            writer.WriteAttributeString("NodeRadius", "100");
                    }
                    // The Outline attribute seems to be ignored.
                    //writer.WriteAttributeString("Outline", colorToString(node.Attr.Color));
                    writer.WriteEndElement();
                }
                writer.WriteEndElement();
                writer.WriteStartElement("Links");
                foreach (GraphViews.Edge edge in edges)
                {
                    writer.WriteStartElement("Link");
                    writer.WriteAttributeString("Source", edge.Source.ID);
                    writer.WriteAttributeString("Target", edge.Target.ID);
                    writer.WriteAttributeString("Label", edge.Label);
                    writer.WriteAttributeString("FontSize", edge.FontSize.ToString() ?? Microsoft.Msagl.Drawing.Label.DefaultFontSize.ToString());
                    // Sadly there is no way to change the arrow head style in DGML.
                    if (edge.UserData is GraphViews.GraphBuilder.Group)
                        writer.WriteAttributeString("Category", "Contains");
                    writer.WriteEndElement();
                }
                writer.WriteEndElement();
                writer.WriteEndElement();
            }
        }
    }

    internal class MsaglWriter : GraphViews.GraphWriter
    {
        public MsaglWriter(string name) : base(name) { }

        public override void Write(string path)
        {
            throw new NotImplementedException();
        }

        public Graph ToGraph()
        {
            Graph g = new Graph(Name);
            foreach(var node in nodes.Values)
            {
                g.AddNode(Convert(node));
            }
            foreach(var edge in edges)
            {
                Edge newEdge = g.AddEdge(edge.Source.ID, edge.Label, edge.Target.ID);
                if (newEdge.Label != null)
                {
                    newEdge.Label.FontSize = edge.FontSize;
                    newEdge.Label.FontColor = Convert(edge.FontColor);
                }
                newEdge.UserData = edge.UserData;
                newEdge.Attr.ArrowheadAtSource = Convert(edge.ArrowheadAtSource);
                newEdge.Attr.ArrowheadAtTarget = Convert(edge.ArrowheadAtTarget);
            }
            return g;
        }

        private ArrowStyle Convert(GraphViews.ArrowheadStyle arrowheadStyle)
        {
            switch(arrowheadStyle)
            {
                case GraphViews.ArrowheadStyle.None: return ArrowStyle.None;
                case GraphViews.ArrowheadStyle.Normal: return ArrowStyle.Normal;
                default: throw new ArgumentException($"Unknown ArrowheadStyle: {arrowheadStyle}", nameof(arrowheadStyle));
            }
        }

        private Node Convert(GraphViews.Node node)
        {
            Node nd = new Node(node.ID);
            nd.LabelText = node.Label;
            nd.Label.FontSize = node.FontSize;
            nd.Label.FontColor = Convert(node.FontColor);
            nd.Attr.Shape = Shape.Box;
            nd.Attr.FillColor = Convert(node.FillColor);
            if(node.Shape == GraphViews.ShapeStyle.None)
                nd.Attr.Color = Color.White;
            return nd;
        }

        private Color Convert(GraphViews.Color color)
        {
            switch(color)
            {
                case GraphViews.Color.Black: return Color.Black;
                case GraphViews.Color.Blue: return Color.Blue;
                case GraphViews.Color.Empty: return Color.Transparent;
                case GraphViews.Color.Green: return Color.Green;
                case GraphViews.Color.LightGray: return Color.LightGray;
                case GraphViews.Color.Purple: return Color.Purple;
                case GraphViews.Color.Red: return Color.Red;
                case GraphViews.Color.White: return Color.White;
                default: throw new ArgumentException($"Unknown Color: {color}", nameof(color));
            }
        }
    }
}