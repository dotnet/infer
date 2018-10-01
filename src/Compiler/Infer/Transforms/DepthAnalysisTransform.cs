// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using System.Text;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    internal class DepthAnalysisTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "DepthAnalysisTransform"; }
        }

        internal class DepthInfo
        {
            public int definitionDepth;
            public int minDepth = int.MaxValue;
            public int useCount;
            public Dictionary<int, IndexInfo> indexInfoOfDepth = new Dictionary<int, IndexInfo>();
        }

        internal class IndexInfo
        {
            public int literalIndexingDepth;

            /// <summary>
            /// The intersection of all containers that the expression has appeared in
            /// </summary>
            internal Containers containers;

            /// <summary>
            /// The variable reference that the expression should be replaced with.
            /// </summary>
            public IExpression clone;

            public override string ToString()
            {
                StringBuilder sb = new StringBuilder("IndexInfo");
                sb.Append("  containers = ");
                sb.AppendLine(containers.ToString());
                return sb.ToString();
            }
        }

        public Dictionary<IVariableDeclaration, DepthInfo> depthInfos = new Dictionary<IVariableDeclaration, DepthInfo>();

        protected override IExpression ConvertVariableDeclExpr(IVariableDeclarationExpression ivde)
        {
            bool isDef = Recognizer.IsBeingMutated(context, ivde);
            if (isDef) RegisterDepth(ivde, isDef);
            return ivde;
        }

        protected override IExpression ConvertVariableRefExpr(IVariableReferenceExpression ivre)
        {
            bool isDef = Recognizer.IsBeingMutated(context, ivre);
            RegisterDepth(ivre, isDef);
            return ivre;
        }

        protected override IExpression ConvertArrayIndexer(IArrayIndexerExpression iaie)
        {
            bool isDef = Recognizer.IsBeingMutated(context, iaie);
            base.ConvertArrayIndexer(iaie);
            RegisterDepth(iaie, isDef);
            return iaie;
        }

        private void RegisterDepth(IExpression expr, bool isDef)
        {
            if (Recognizer.IsBeingIndexed(context)) return;
            int depth = Recognizer.GetIndexingDepth(expr);
            IVariableDeclaration baseVar = Recognizer.GetVariableDeclaration(expr);
            // If not an indexed variable reference, skip it (e.g. an indexed argument reference)
            if (baseVar == null) return;
            // If the variable is not stochastic, skip it
            if (!CodeRecognizer.IsStochastic(context, baseVar)) return;
            ChannelInfo ci = context.InputAttributes.Get<ChannelInfo>(baseVar);
            if (ci != null && ci.IsMarginal) return;
            DepthInfo depthInfo;
            if (!depthInfos.TryGetValue(baseVar, out depthInfo))
            {
                depthInfo = new DepthInfo();
                depthInfos[baseVar] = depthInfo;
            }
            if (isDef)
            {
                depthInfo.definitionDepth = depth;
                return;
            }
            depthInfo.useCount++;
            if (depth < depthInfo.minDepth) depthInfo.minDepth = depth;
            int literalIndexingDepth = 0;
            foreach (var bracket in Recognizer.GetIndices(expr))
            {
                if (!bracket.All(index => index is ILiteralExpression)) break;
                literalIndexingDepth++;
            }
            IndexInfo info;
            if (depthInfo.indexInfoOfDepth.TryGetValue(depth, out info))
            {
                Containers containers = new Containers(context);
                info.containers = Containers.Intersect(info.containers, containers);
                info.literalIndexingDepth = System.Math.Min(info.literalIndexingDepth, literalIndexingDepth);
            }
            else
            {
                info = new IndexInfo();
                info.containers = new Containers(context);
                info.literalIndexingDepth = literalIndexingDepth;
                depthInfo.indexInfoOfDepth[depth] = info;
            }
        }

        protected override IExpression ConvertMethodInvoke(IMethodInvokeExpression imie)
        {
            // do not count argument of Infer as a use
            if (CodeRecognizer.IsInfer(imie)) return imie;
            return base.ConvertMethodInvoke(imie);
        }
    }
}