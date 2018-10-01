// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Reflection;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Replaces references to the parameters of a method with actual literal values.
    /// The parameters of the method are then removed.
    /// </summary>
    internal class ParameterInsertionTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "ParameterInsertionTransform"; }
        }

        public MethodBase MethodToTransform;
        private IList<object> args;

        public ParameterInsertionTransform(MethodBase MethodToTransform, IList<object> args)
        {
            this.MethodToTransform = MethodToTransform;
            this.args = args;
        }

        protected override IMethodDeclaration ConvertMethod(IMethodDeclaration imd)
        {
            MethodBase mb = Builder.ToMethod(imd);
            if (mb != MethodToTransform) return null;
            IMethodDeclaration md = (IMethodDeclaration) base.ConvertMethod(imd);
            md.Parameters.Clear();
            return md;
        }

        protected override IExpression ConvertArgumentRef(IArgumentReferenceExpression iare)
        {
            IParameterDeclaration ipd = iare.Parameter.Resolve();
            IMethodDeclaration imd = context.FindAncestor<IMethodDeclaration>();
            int index = imd.Parameters.IndexOf(ipd);
            if (index == -1)
            {
                Error("Parameter value not found for '" + iare.Parameter.Name + "'.");
                return iare;
            }
            return Builder.LiteralExpr(args[index]);
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}