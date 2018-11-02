// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
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
using Microsoft.ML.Probabilistic.Compiler.Attributes;

namespace Microsoft.ML.Probabilistic.Compiler.Visualizers
{
    /// <summary>
    /// Interaction logic for TransformerView.xaml
    /// </summary>
    public partial class TransformerView : UserControl
    {
        public TransformerView()
        {
            InitializeComponent();
            // This code makes it so that searches only happen after the user hasn't typed for 250ms
            var cel = new ConsolidatingEventListener<TextChangedEventArgs>(250, SearchTextBox_TextChanged);
            SearchTextBox.TextChanged += new TextChangedEventHandler(cel.HandlerMethod);
            MyDeclarationView.MyListView.SelectionChanged += SelectionChanged;
        }

        void SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (e.AddedItems.Count == 0)
                return;
            var item = e.AddedItems[0];
            LineOfCode line;
            if (item is LineOfCode)
                line = (LineOfCode)item;
            else if (item is FaintLine)
                line = ((FaintLine)item).Line;
            else if (item is MatchedLine)
                line = ((MatchedLine)item).Line;
            else
                line = null;
            CodeElementInspector.DataContext = line;
        }

        Dictionary<DebugInfo, TabItem> tabItems = new Dictionary<DebugInfo, TabItem>();

        private void UserControl_DataContextChanged(object sender, DependencyPropertyChangedEventArgs e)
        {
            var transformer = DataContext as CodeTransformer;
            if (transformer == null || transformer.transformMap.Count == 0) return;
            ITypeDeclaration itd = transformer.transformMap.Values.First();
            if (itd == null)
                return;
            MyDeclarationView.DataContext = itd;
            var attributes = transformer.Transform.Context.InputAttributes;

            MyTabControl.Items.Clear();
            TabItem firstItem = new TabItem();
            firstItem.Content = MyDeclarationView;
            MyTabControl.Items.Add(firstItem);
            MyTabControl.SelectedIndex = 0;
            // if the output has DebugInfo attributes attached, display these as alternate tabs in the TabControl
            var infos = attributes.GetAll<DebugInfo>(itd);
            foreach (var info in infos)
            {
                if (info.Transform != transformer.Transform)
                    continue;
                TabItem item;
                if (!tabItems.TryGetValue(info, out item))
                {
                    item = new TabItem();
                    item.Header = info.Name;
                    if (info.Value is ITypeDeclaration || info.Value is IStatement || info.Value is Func<SourceNode>)
                    {
                        var view = new DeclarationView();
                        view.MyListView.SelectionChanged += SelectionChanged;
                        view.DataContext = info.Value;
                        item.Content = view;
                    }
                    tabItems[info] = item;
                }
                MyTabControl.Items.Add(item);
            }
            if (MyTabControl.Items.Count > 1)
                firstItem.Header = "Output";

            CodeElementInspector.attributes = attributes;
            // MyDeclarationView will fill in this dictionary for the CodeElementInspector to use
            CodeElementInspector.parentOfLine = MyDeclarationView.parentOfLine;
            var errors = transformer.Transform.Context.Results.ErrorsAndWarnings;
            ErrorsTable.ItemsSource = errors;
            bool showErrors = errors.Count>0;
            ErrorsSplitter.Visibility = showErrors ? Visibility.Visible: Visibility.Collapsed;
            ErrorsRow.Height = showErrors ? new GridLength(1,GridUnitType.Star) : new GridLength(0);
        }

        CodeSearchQuery query;

        private void SearchTextBox_TextChanged(object sender, TextChangedEventArgs e)
        {
            string searchText = SearchTextBox.Text;
            if (string.IsNullOrWhiteSpace(searchText))
            {
                query = null;
                ClearSearch.Visibility = Visibility.Collapsed;
            }
            else
            {
                query = CodeSearchQuery.FromQueryString(searchText);
                ClearSearch.Visibility = Visibility.Visible;
            }
            SetQuery(query);
        }

        private void SetQuery(CodeSearchQuery query)
        {
            TabItem item = (TabItem)MyTabControl.SelectedItem;
            if (item == null)
                return;
            if (item.Content is DeclarationView)
                ((DeclarationView)item.Content).Query = query;
        }

        private void SearchTextBox_MouseDoubleClick(object sender, MouseButtonEventArgs e)
        {
            SearchTextBox.SelectionStart = 0;
            SearchTextBox.SelectionLength = SearchTextBox.Text.Length;
        }

        private void ClearSearch_Click(object sender, RoutedEventArgs e)
        {
            SearchTextBox.Text = "";
            SearchTextBox.Focus();
        }

        private void CommandBinding_Executed_1(object sender, ExecutedRoutedEventArgs e)
        {
            var s = e.Parameter as string;
            var q2 = CodeSearchQuery.FromQueryString(SearchTextBox.Text);
            q2.AddTerm("\""+s+"\"");
            SearchTextBox.Text = q2.ToQueryString();
        }

        private void ErrorsTable_SelectedCellsChanged(object sender, SelectedCellsChangedEventArgs e)
        {
            var te = ErrorsTable.SelectedItem as TransformError;
            if (te == null)
                return;
            MyDeclarationView.SelectCodeElement(te.DisplayTag);
        }

        private void CloseAttributePanel(object sender, ExecutedRoutedEventArgs e)
        {
            AttributesColumn.Width = new GridLength(0);
        }

        private void OpenAttributePanel(object sender, ExecutedRoutedEventArgs e)
        {
            if (AttributesColumn.ActualWidth <10)
            {
                AttributesColumn.Width = new GridLength(250);
            }
        }

        private void MyTabControl_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            DeclarationView view = null;
            TabItem item = (TabItem)MyTabControl.SelectedItem;
            if (item == null)
                return;
            if (item.Content is DeclarationView)
                view = (DeclarationView)item.Content;
            if(view != null)
                CodeElementInspector.parentOfLine = view.parentOfLine;
            SetQuery(query);
        }
    }

    /// <summary>
    /// Converts a TransformError to various values for the error table, depending on the parameter.
    /// </summary>
    internal class TransformErrorConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            var err = value as TransformError;
            if (err == null) return null;
            if ("Type".Equals(parameter))
            {
                return err.IsWarning ? "Warning" : "Error";
            }
            if ("InputElement".Equals(parameter))
            {
                string elementString = ToStringSafe(err.InputElement);
                if (elementString.Length > TransformError.maxElementStringLength)
                    elementString = elementString.Remove(TransformError.maxElementStringLength);
                return elementString;
            }
            if ("Exception".Equals(parameter))
            {
                var s = "-";
                if (err.exception != null) s = err.exception.Message;
                return s;
            }
            return null;
        }

        private string ToStringSafe(object obj)
        {
            try
            {
                if (obj == null) return "null";
                return obj.ToString();
            }
            catch (Exception ex)
            {
                return "Could not evaluate ToString(): " + ex;
            }
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}
