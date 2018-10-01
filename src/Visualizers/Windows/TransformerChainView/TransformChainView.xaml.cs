// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using Microsoft.ML.Probabilistic.Compiler;
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

namespace Microsoft.ML.Probabilistic.Compiler.Visualizers
{
    /// <summary>
    /// The top-level control for the transform browser.  It contains a TransformerView that shows the selected transform.
    /// The TransformerView contains a DeclarationView and a CodeElementInspector.
    /// </summary>
    public partial class TransformChainView : UserControl
    {
        public TransformChainView()
        {
            InitializeComponent();
        }

        private void UserControl_DataContextChanged(object sender, DependencyPropertyChangedEventArgs e)
        {
            var tc = DataContext as TransformerChain;

            // Add active transformation steps to the list
            var l = new List<CodeTransformer>();
            foreach (CodeTransformer ct in tc.transformers)
            {
                if (ct.OutputEqualsInput && ct.Transform.Context.Results.IsSuccess) continue;
                l.Add(ct);
            }

            // Show in the list box
            MyListBox.ItemsSource = l;
            MyListBox.SelectedIndex = 0;
        }

        /// <summary>
        /// Show this view in a window.
        /// </summary>
        /// <param name="title"></param>
        /// <param name="tcv"></param>
        /// <param name="maximise"></param>
        internal static void ShowInWindow(string title, TransformChainView tcv, bool maximise)
        {
            Window w = new Window
            {
                Title = title,
                Width = 1024,
                Height = 800,
            };
            w.Content = tcv;
            w.ShowDialog();
            if (maximise) w.WindowState = WindowState.Maximized;
        }

        /// <summary>
        /// The currently selected code transformer.
        /// </summary>
        internal CodeTransformer SelectedTransformer
        {
            get { return MyListBox.SelectedItem as CodeTransformer; }
            set { MyListBox.SelectedItem = value; }
        }

        private void MyListBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            MyTransformerView.DataContext = MyListBox.SelectedItem; 
        }
    }

    /// <summary>
    /// Converts a CodeTransformer to various visual property values, depending on the parameter.
    /// </summary>
    internal class TransformConverter : IValueConverter
    {
        static Brush transparentRed = new SolidColorBrush(Color.FromArgb(128, 255, 0, 0));

        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            var tc = (CodeTransformer)value;
            if ("Text".Equals(parameter))
            {
                return tc.GetFriendlyName();
            }
            bool isError = tc.Transform.Context.Results.ErrorCount>0;
            if ("Foreground".Equals(parameter))
            {
                return isError ? transparentRed : Brushes.Transparent;
            }
            return null;
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}
