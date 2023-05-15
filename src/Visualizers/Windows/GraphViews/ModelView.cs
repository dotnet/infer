// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using System.Windows.Forms;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.Msagl.GraphViewerGdi;
using Microsoft.Msagl.Drawing;

namespace Microsoft.ML.Probabilistic.Compiler.Visualizers
{
    internal class ModelView : IDisposable
    {
        private readonly GViewer gviewer = new GViewer();

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

        public void Dispose()
        {
            gviewer.Dispose();
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