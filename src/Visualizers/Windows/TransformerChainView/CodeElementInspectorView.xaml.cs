// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using Microsoft.ML.Probabilistic.Utilities;
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
using Microsoft.ML.Probabilistic.Compiler;

namespace Microsoft.ML.Probabilistic.Compiler.Visualizers
{
    /// <summary>
    /// Interaction logic for CodeElementInspectorView.xaml
    /// </summary>
    public partial class CodeElementInspectorView : UserControl
    {
        internal AttributeRegistry<object, ICompilerAttribute> attributes;
        internal Dictionary<LineOfCode, LineOfCode> parentOfLine;

        public CodeElementInspectorView()
        {
            InitializeComponent();
        }

        private void UserControl_DataContextChanged(object sender, DependencyPropertyChangedEventArgs e)
        {
            List<ICompilerAttribute> attrs = new List<ICompilerAttribute>();
            if(DataContext is LineOfCode) 
                ForEachAttributeOfCodeElementAndAncestors((LineOfCode)DataContext, attrs.Add);
            // Ordered by name - good idea?
            AttributeListBox.ItemsSource = attrs.OrderBy(a=>a.GetType().Name);
        }

        private void ForEachAttributeOfCodeElementAndAncestors(LineOfCode line, Action<ICompilerAttribute> action)
        {
            ForEachAttributeOfCodeElementAndContents(line.CodeElement, action);
            if (parentOfLine != null)
            {
                while (line != null && parentOfLine.TryGetValue(line, out line) && !(line.CodeElement is ITypeDeclaration))
                {
                    ForEachAttributeOfCodeElement(line.CodeElement, action);
                }
            }
        }

        private void ForEachAttributeOfCodeElementAndContents(object codeElement, Action<ICompilerAttribute> action)
        {
            if ((attributes == null) || (codeElement==null)) return;
            foreach (var attr in attributes.GetAll<ICompilerAttribute>(codeElement))
                action(attr);
            if (codeElement is IStatement)
            {
                if (codeElement is IExpressionStatement)
                {
                    IExpressionStatement ies = (IExpressionStatement)codeElement;
                    ForEachAttributeOfCodeElement(ies.Expression, action);
                    if (ies.Expression is IVariableDeclarationExpression)
                    {
                        ForEachAttributeOfCodeElement(((IVariableDeclarationExpression)ies.Expression).Variable, action);
                    }
                    else if (ies.Expression is IAssignExpression)
                    {
                        IAssignExpression iae = (IAssignExpression)ies.Expression;
                        if (iae.Target is IVariableDeclarationExpression)
                        {
                            ForEachAttributeOfCodeElement(((IVariableDeclarationExpression)iae.Target).Variable, action);
                        }
                        if (iae.Expression is IMethodInvokeExpression)
                        {
                            ForEachAttributeOfCodeElement(iae.Expression, action);
                        }
                    }
                }
                else if (codeElement is IForStatement)
                {
                    IForStatement ifs = (IForStatement)codeElement;
                    ForEachAttributeOfCodeElementAndContents(ifs.Initializer, action);
                    IBinaryExpression ibe = (IBinaryExpression)ifs.Condition;
                    ForEachAttributeOfCodeElementAndContents(ibe.Right, action);
                }
            }
            if (codeElement is IMethodDeclaration)
            {
                IMethodDeclaration imd = (IMethodDeclaration)codeElement;
                foreach (IParameterDeclaration ipd in imd.Parameters)
                {
                    ForEachAttributeOfCodeElement(ipd, action);
                }
            }
        }

        private void ForEachAttributeOfCodeElement(object codeElement, Action<ICompilerAttribute> action)
        {
            if ((attributes == null) || (codeElement == null))
                return;
            foreach (var attr in attributes.GetAll<ICompilerAttribute>(codeElement))
                action(attr);
        }    
    }

    internal class AttributeToTextConverter : IValueConverter
    {
        public object Convert(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            return StringUtil.ToString(value);
        }

        public object ConvertBack(object value, Type targetType, object parameter, System.Globalization.CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}
