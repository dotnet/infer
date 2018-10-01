// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using Microsoft.ML.Probabilistic.Collections;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
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

namespace Microsoft.ML.Probabilistic.Compiler.Visualizers
{
    /// <summary>
    /// Interaction logic for LineOfCodeView.xaml
    /// </summary>
    public partial class LineOfCodeView : UserControl
    {
        public LineOfCodeView()
        {
            InitializeComponent();
        }
        
        private void UserControl_DataContextChanged(object sender, DependencyPropertyChangedEventArgs e)
        {
            LineOfCode loc;
            MatchCollection matches = null;
            if (DataContext is LineOfCode)
            {
                loc = (LineOfCode)DataContext;
            }
            else if (DataContext is FaintLine)
            {
                loc = ((FaintLine)DataContext).Line;
            }
            else if (DataContext is MatchedLine)
            {
                MatchedLine ml = (MatchedLine)DataContext;
                loc = ml.Line;
                matches = ml.Matches;
            }
            else
                return;
            CodeTextBlock.Inlines.Clear();
            SyntaxHighlighting.PrintWithVerySimpleSyntaxHighlighting(CodeTextBlock, loc.Text, matches);
        }
    }

    public static class SyntaxHighlighting
    {
        private static Regex reg = new Regex(@"[#\w]+");

        /// <summary>
        /// Very simple syntax highlighting (copied from the examples browser)
        /// </summary>
        /// <param name="tb"></param>
        /// <param name="s"></param>
        /// <param name="matches"></param>
        internal static void PrintWithVerySimpleSyntaxHighlighting(TextBlock tb, string s, MatchCollection matches)
        {
            bool doHighlighting = (s.Length < 1000);  // don't highlight very long lines since they crash the UI
            if (!doHighlighting)
            {
                tb.Inlines.Add(new Run(s));
                return;
            }
            // todo: Use the actual AST to do syntax highlighting
            bool isComment = s.Trim().StartsWith("//");
            var foreground = isComment ? Brushes.Green : Brushes.Black;
            MatchCollection mc = reg.Matches(s);
            int ind = 0;
            foreach (Match m in mc)
            {
                var brush = foreground;
                tb.Inlines.Add(new Run(s.Substring(ind, m.Index - ind))
                {
                    Foreground = brush
                });
                ind = m.Index + m.Length;
                string word = s.Substring(m.Index, m.Length);
                if (!isComment)
                {
                    bool reserved = IsReservedWord(word);
                    if (reserved)
                    {
                        tb.Inlines.Add(new Run
                        {
                            Text = word,
                            Foreground = Brushes.Blue
                        });
                        continue;
                    }
                    if (IsKnownTypeName(word))
                        brush = Brushes.DarkCyan;
                }
                var background = Overlaps(matches, m.Index, m.Length) ? Brushes.LightGray : Brushes.Transparent;
                var hp = new Hyperlink
                {
                    Foreground = brush,
                    Background = background,
                    CommandParameter = word,
                    Command = NavigationCommands.Search,
                    FontFamily = tb.FontFamily,
                    Cursor = Cursors.Hand
                };
                hp.Inlines.Add(new Run(word));
                tb.Inlines.Add(hp);
            }
            if (ind < s.Length)
                tb.Inlines.Add(new Run(s.Substring(ind))
                {
                    Foreground = foreground
                });
        }

        private static bool Overlaps(MatchCollection matches, int start, int count)
        {
            if (matches == null)
                return false;
            foreach (Match match in matches)
            {
                if (match.Index <= start)
                {
                    if (match.Length + match.Index > start)
                        return true;
                }
                else
                {
                    if (start + count > match.Index)
                        return true;
                }
            }
            return false;
        }

        private static Set<string> knownTypes = new Set<string>();

        private static bool IsKnownTypeName(string s)
        {
            if (knownTypes.Count == 0)
            {
                knownTypes.AddRange(new string[]
                    {
                        "Console", "List", "Dictionary", "Set",
                        "Variable", "Range", "InferenceEngine", "VariableArray", "IVariableArray", "ExpectationPropagation", "VariationalMessagePassing", 
                        "Vector", "Matrix", "PositiveDefiniteMatrix", "Rand", 
                        "VectorGaussian", "Gaussian", "Gamma", "Bernoulli", "Discrete", "Beta", "Dirichlet", "Binomial", "Poisson", "Wishart",
                        "InferNet", "Factor", "QueryTypes"
                    });
            }
            return knownTypes.Contains(s);
        }

        private static Dictionary<string, bool> reservedSet = new Dictionary<string, bool>();

        private static string[] RESERVED_WORDS =
            {
                "abstract", "as", "base", "bool", "break", "byte", "case", "catch", "char", "checked",
                "class", "const", "continue", "decimal", "default", "delegate", "do", "double", "else", "enum", "event", "explicit", "extern", "false",
                "finally", "fixed", "float", "for", "foreach", "goto", "if", "implicit", "in", "int", "interface", "internal", "is", "lock", "long", "namespace",
                "new", "null", "object", "operator", "out", "override", "params", "private", "protected", "public", "readonly", "ref", "return", "sbyte",
                "sealed", "short", "sizeof", "stackalloc", "static", "string", "struct", "switch", "this", "throw", "true", "try", "typeof", "uint", "ulong",
                "unchecked", "unsafe", "ushort", "using", "virtual", "volatile", "void", "while","#region","#endregion"
            };

        private static bool IsReservedWord(string word)
        {
            if (reservedSet.Count == 0)
            {
                foreach (string s in RESERVED_WORDS) reservedSet[s] = true;
            }
            return reservedSet.ContainsKey(word);
        }
    }
}
