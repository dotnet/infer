// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.Graphs;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using System.IO;
using System.Windows.Forms;
#if INCLUDE_TRANSFORM_BROWSER
using System.Windows.Threading;
#endif

namespace Microsoft.ML.Probabilistic.Compiler.Visualizers
{
    public class WindowsVisualizer : Visualizer
    {
        internal override IFactorManager FactorManager { get; } = new DefaultVisualizer().FactorManager;

        internal override IDependencyGraphVisualizer DependencyGraphVisualizer { get; } = new FormsDependencyGraphVisualizer();

        internal override ITransformerChainVisualizer TransformerChainVisualizer { get; } = new WPFTransformerChainVisualizer();

        internal override ITaskGraphVisualizer TaskGraphVisualizer { get; } = new FormsTaskGraphVisualizer();

        internal override IFactorGraphVisualizer FactorGraphVisualizer { get; } = new FormsFactorGraphVisualizer();

        internal override IGraphWriter GraphWriter { get; } = new DefaultVisualizer().GraphWriter;
        
        public static class FormHelper
        {
            private static System.Drawing.Icon icon;

            public static System.Drawing.Icon Icon
            {
                get
                {
                    if (icon == null)
                    {
                        // If this fails, debug with:
                        // typeof(ModelView).Assembly.GetManifestResourceNames()
                        Stream iconStream = typeof(InferenceEngine).Assembly.GetManifestResourceStream("Microsoft.ML.Probabilistic.Infer.Infer.ico");
                        icon = new System.Drawing.Icon(iconStream);
                        //icon = System.Drawing.Icon.ExtractAssociatedIcon("Infer.ico");
                    }
                    return icon;
                }
            }

            public static Form ShowInForm(Control c, string title, bool maximise)
            {
                Form f = new Form();
                f.Text = title;
                f.Icon = Icon;
                f.Size = new System.Drawing.Size(1000, 800);
                c.Dock = DockStyle.Fill;
                f.Controls.Add(c);
                f.Show();
                f.BringToFront();
                if (maximise) f.WindowState = FormWindowState.Maximized;
                return f;
            }

            public static void RunInForm(Control c, string title, bool maximise)
            {
                Application.EnableVisualStyles();
                Application.Run(ShowInForm(c, title, maximise));
            }
        }

        internal class WPFTransformerChainVisualizer : ITransformerChainVisualizer
        {
#if INCLUDE_TRANSFORM_BROWSER
            // When this object is constructed, we save the dispatcher of the UI thread.
            readonly Dispatcher dispatcher = Dispatcher.CurrentDispatcher;
#endif

            public void VisualizeTransformerChain(TransformerChain target, string folder, string modelName)
            {
#if INCLUDE_TRANSFORM_BROWSER
                int sel = 0;
                for (int i = 0; i < target.transformers.Count; i++)
                {
                    TransformResults tr = target.transformers[i].Transform.Context.Results;
                    if (tr.ErrorCount > 0)
                    {
                        sel = i;
                        break;
                    }
                }
                // We may not be on the UI thread right now, so we call the saved dispatcher.
                dispatcher.Invoke(() =>
                {
                    TransformChainView tcv = new TransformChainView();
                    tcv.DataContext = target;
                    tcv.SelectedTransformer = target.transformers[sel];
                    TransformChainView.ShowInWindow(modelName + " transform chain", tcv, false);
                });
#endif
            }
        }

        internal class FormsDependencyGraphVisualizer : IDependencyGraphVisualizer
        {
            public void VisualizeDependencyGraph(IndexedGraph dg, IEnumerable<EdgeStylePredicate> edgeStyles = null, Func<int, string> nodeName = null, Func<int, string> edgeName = null, string visualizationTitle = "Infer.NET Dependency Graph Viewer")
            {
                var view = new DependencyGraphView(dg, edgeStyles, nodeName, edgeName);
                view.RunInForm(visualizationTitle);
            }
        }

        internal class FormsTaskGraphVisualizer : ITaskGraphVisualizer
        {
            public void VisualizeTaskGraph(ITypeDeclaration itd, BasicTransformContext context)
            {
                var view = new TaskGraphView(itd, context);
                view.RunInForm();
            }
        }

        internal class FormsFactorGraphVisualizer : IFactorGraphVisualizer
        {
            public void VisualizeFactorGraph(ModelBuilder model)
            {
                ModelView modelView = new ModelView(model);
                modelView.ShowInForm(model.modelType.Name, false);
            }
        }
    }
}
