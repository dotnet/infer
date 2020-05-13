// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Utilities;
using System.Linq;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Convert occurrences of <c>IsIncreasing(i)</c> into literal boolean constants.
    /// </summary>
    internal class IsIncreasingTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get
            {
                return nameof(IsIncreasingTransform);
            }
        }

        HashSet<string> backwardLoops = new HashSet<string>();

        protected override IStatement ConvertFor(IForStatement ifs)
        {
            string toRemove = null;
            if(!Recognizer.IsForwardLoop(ifs))
            {
                var loopVar = Recognizer.LoopVariable(ifs);
                toRemove = loopVar.Name;
                backwardLoops.Add(toRemove);
            }
            var result = base.ConvertFor(ifs);
            if (toRemove != null) backwardLoops.Remove(toRemove);
            return result;
        }

        protected override IExpression ConvertMethodInvoke(IMethodInvokeExpression imie)
        {
            if(CodeRecognizer.IsIsIncreasing(imie))
            {
                IExpression arg = imie.Arguments[0];
                IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(arg);
                if (backwardLoops.Contains(ivd.Name)) return Builder.LiteralExpr(false);
                else return Builder.LiteralExpr(true);
            }
            return base.ConvertMethodInvoke(imie);
        }
    }
}
