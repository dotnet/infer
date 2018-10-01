// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler;

namespace Microsoft.ML.Probabilistic.Compiler.Attributes
{
    /// <summary>
    /// Stores debugging information to show in the transform browser.  The browser will create a tab for each DebugInfo object.
    /// This attribute should be attached to the top-level ITypeDeclaration produced by a transform.
    /// </summary>
    internal class DebugInfo : ICompilerAttribute
    {
        /// <summary>
        /// The name of the tab in the browser.
        /// </summary>
        public string Name;
        /// <summary>
        /// A DataContext for DeclarationView.  Currently this must be a code object (ITypeDeclaration or IStatement or Func&lt;SourceNode&gt;).
        /// </summary>
        public object Value;
        /// <summary>
        /// The transform in the browser that will show this tab.
        /// </summary>
        public ICodeTransform Transform;

        public override string ToString()
        {
            return string.Format("DebugInfo({0},{1})", Transform == null ? "" : Transform.Name, Name);
        }
    }
}
