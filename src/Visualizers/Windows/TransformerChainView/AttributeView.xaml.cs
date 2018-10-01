// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
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
    /// Interaction logic for AttributeView.xaml
    /// </summary>
    public partial class AttributeView : UserControl
    {
        public AttributeView()
        {
            InitializeComponent();
        }

        private void UserControl_DataContextChanged(object sender, DependencyPropertyChangedEventArgs e)
        {
            var text = DataContext as string;
            if (text == null)
                return;
            AttributeTextBlock.Inlines.Clear();
            SyntaxHighlighting.PrintWithVerySimpleSyntaxHighlighting(AttributeTextBlock, text, default(MatchCollection));
        }
    }
}
