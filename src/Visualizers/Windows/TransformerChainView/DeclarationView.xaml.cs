// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Text.RegularExpressions;
using Microsoft.ML.Probabilistic.Compiler;

namespace Microsoft.ML.Probabilistic.Compiler.Visualizers
{
    /// <summary>
    /// Interaction logic for DeclarationView.xaml
    /// </summary>
    public partial class DeclarationView : UserControl
    {
        internal LanguageWriter languageWriter = new CSharpWriter();
        /// <summary>
        /// Used by CodeElementInspector.
        /// </summary>
        internal Dictionary<LineOfCode, LineOfCode> parentOfLine = new Dictionary<LineOfCode, LineOfCode>();

        CodeSearchQuery query;
        /// <summary>
        /// The current search query (or null if none)
        /// </summary>
        internal CodeSearchQuery Query {
            get { return query; }
            set
            {
                if (query != value)
                {
                    query = value;
                    QueryHasChanged();
                }
            }
        }

        public DeclarationView()
        {
            InitializeComponent();
        }

        List<LineOfCode> lines = new List<LineOfCode>();
        private void UserControl_DataContextChanged(object sender, DependencyPropertyChangedEventArgs e)
        {
            lines = new List<LineOfCode>();
            parentOfLine.Clear();
            SourceNode sourceNode;
            if (DataContext is ITypeDeclaration)
                sourceNode = languageWriter.GeneratePartialSource((ITypeDeclaration)DataContext);
            else if (DataContext is IStatement)
                sourceNode = languageWriter.GeneratePartialSource((IStatement)DataContext);
            else if (DataContext is Func<SourceNode>)
                sourceNode = ((Func<SourceNode>)DataContext)();
            else
                return;
            AddLines(sourceNode);
            QueryHasChanged();
        }

        private void QueryHasChanged()
        {
            if (Query == null)
            {
                MyListView.ItemsSource = lines;
            }
            else
            {
                var list = new List<object>();
                CollapsedCode collapsed = null;
                foreach (var l in lines)
                {
                    var matches = Query.Matches(l);
                    if (matches.Count > 0)
                    {
                        if (collapsed != null)
                        {
                            AddCollapsed(list, collapsed);
                            collapsed = null;
                        }
                        list.Add(new MatchedLine()
                        {
                            Matches = matches,
                            Line = l
                        });
                    }
                    else
                    {
                        if (collapsed == null) collapsed = new CollapsedCode { CollapsedLines = new List<LineOfCode>() };
                        collapsed.CollapsedLines.Add(l);
                    }
                }
                if (collapsed != null) AddCollapsed(list, collapsed); 
                MyListView.ItemsSource = list;
                MyListView.Tag = Query;
            }
        }

        int MIN_LINES_TO_COLLAPSE = 0;
        private void AddCollapsed(List<object> list, CollapsedCode collapsed)
        {
            if (collapsed.CollapsedLines.Count <= MIN_LINES_TO_COLLAPSE)
            {
                list.AddRange(collapsed.CollapsedLines.Select(l => new FaintLine { Line = l }));
            }
            else
            {
                if(MIN_LINES_TO_COLLAPSE > 0)
                    list.Add(new FaintLine { Line = collapsed.CollapsedLines.First() });
                list.Add(collapsed);
                if(MIN_LINES_TO_COLLAPSE > 0)
                    list.Add(new FaintLine { Line = collapsed.CollapsedLines.Last() });
            }
        }

        private void AddLines(SourceNode sourceNode)
        {
            AddLines(sourceNode, line =>
            {
            });
        }

        private void AddLines(SourceNode sourceNode, Action<LineOfCode> action)
        {
            LineOfCode parentLine = null;
            AddLines(sourceNode, sourceNode.StartString, delegate(LineOfCode line)
            {
                action(line);
                parentLine = line;
            });
            if (sourceNode.Children != null)
            {
                foreach (var child in sourceNode.Children)
                {
                    AddLines(child, delegate(LineOfCode line)
                    {
                        if (parentLine == null)
                            action(line);
                        else // note the parent action is not executed here
                            parentOfLine[line] = parentLine;
                    });
                }
            }
            AddLines(sourceNode, sourceNode.EndString, action);                    
        }

        char[] lineBreaks = { '\r', '\n' };
        private void AddLines(SourceNode sn, string text, Action<LineOfCode> action)
        {
            string[] ls = text.Split(lineBreaks, StringSplitOptions.RemoveEmptyEntries);
            foreach (var s in ls)
            {
                var line = new LineOfCode
                {
                    Text = s,
                    CodeElement = sn.ASTElement
                };
                action(line);
                lines.Add(line);
            }
        }

        internal void SelectCodeElement(object codeElement)
        {
            // todo: this cannot work right now, since we want to look at the input elements, not the output ones.
            var selectedLine = lines.FirstOrDefault(l => ReferenceEquals(l.CodeElement, codeElement));
            if (selectedLine == null) return;
            // todo: make this line visible, if it is collapsed
            MyListView.SelectedItem = selectedLine;
            MyListView.ScrollIntoView(selectedLine);
        }

        private void CopyCode_Click(object sender, RoutedEventArgs e)
        {
            var sb = new StringBuilder();
            foreach (var l in lines) sb.AppendLine(l.Text);
            Clipboard.SetText(sb.ToString());
        }
    }

    /// <summary>
    /// Represents a line of source code
    /// </summary>
    internal class LineOfCode
    {
        internal string Text { get; set; }
        internal object CodeElement {get;set;}

        public override string ToString()
        {
            return Text;
        }
    }

    /// <summary>
    /// View model for a line of code with query match information
    /// </summary>
    internal class MatchedLine
    {
        internal MatchCollection Matches;
        internal LineOfCode Line
        {
            get;
            set;
        }
    }

    /// <summary>
    /// View model for a 'faint' i.e. faded out, line of code
    /// </summary>
    internal class FaintLine
    {
        internal LineOfCode Line { get; set; }
    }

    /// <summary>
    /// View model for a collapsed block of code
    /// </summary>
    internal class CollapsedCode
    {
        internal List<LineOfCode> CollapsedLines { get; set; }
    }
}
