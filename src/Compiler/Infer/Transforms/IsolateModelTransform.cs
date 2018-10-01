// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Reflection;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Transforms the code to contain a single method.
    /// </summary>
    internal class IsolateModelTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "IsolateModelTransform"; }
        }

        public MethodBase MethodToTransform;

        public IsolateModelTransform(MethodBase MethodToTransform)
        {
            this.MethodToTransform = MethodToTransform;
        }

        protected override IMethodDeclaration ConvertMethod(IMethodDeclaration imd)
        {
            if (MethodToTransform != null)
            {
                MethodBase mb = Builder.ToMethod(imd);
                if (mb != MethodToTransform) return null;
            }
            return base.ConvertMethod(imd);
        }
    }
}