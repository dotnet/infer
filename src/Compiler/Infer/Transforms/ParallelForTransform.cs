// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using System.Threading.Tasks;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Converts for loops into Parallel.For() calls.
    /// </summary>
    internal class ParallelForTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "ParallelForTransform"; }
        }

        /// <summary>
        /// Converts for loops into parallel for loops.
        /// </summary>
        /// <param name="ifs">The for loop to convert</param>
        /// <returns>The converted statement</returns>
        protected override IStatement ConvertFor(IForStatement ifs)
        {
            if (context.InputAttributes.Has<ConvergenceLoop>(ifs) || context.InputAttributes.Has<HasOffsetIndices>(ifs) || (ifs is IBrokenForStatement))
                return base.ConvertFor(ifs);
            IVariableDeclaration loopVar = Recognizer.LoopVariable(ifs);
            if (context.InputAttributes.Has<Sequential>(loopVar)) return base.ConvertFor(ifs);
            // Convert this loop to a parallel for loop
            IExpression loopSize = null;
            IExpression loopStart = null;
            if (Recognizer.IsForwardLoop(ifs))
            {
                loopStart = Recognizer.LoopStartExpression(ifs);
                loopSize = Recognizer.LoopSizeExpression(ifs);
            }
            else if (ifs.Condition is IBinaryExpression ibe) 
            {
                if (ibe.Operator == BinaryOperator.GreaterThanOrEqual)
                {
                    // loop is "for(int i = end; i >= start; i--)"
                    loopStart = ibe.Right;
                    loopSize = Builder.BinaryExpr(Recognizer.LoopStartExpression(ifs), BinaryOperator.Add, Builder.LiteralExpr(1));
                }
            }
            if (loopSize == null) return base.ConvertFor(ifs);
            IAnonymousMethodExpression bodyDelegate = Builder.AnonMethodExpr(typeof (Action<int>));
            bodyDelegate.Body = ifs.Body;
            bodyDelegate.Parameters.Add(Builder.Param(loopVar.Name, loopVar.VariableType));
            Delegate d = new Func<int, int, Action<int>, ParallelLoopResult>(Parallel.For);
            IMethodInvokeExpression parallelFor = Builder.StaticMethod(d, loopStart, loopSize, bodyDelegate);
            IStatement st = Builder.ExprStatement(parallelFor);
            return st;
        }

        /// <summary>
        /// Leaves expression statements (including existing Parallel.For loops) unchanged.
        /// </summary>
        /// <param name="ies"></param>
        /// <returns></returns>
        protected override IStatement ConvertExpressionStatement(IExpressionStatement ies)
        {
            return ies;
        }
    }

    /// <summary>
    /// Attached to loops to indicate that some sub-statement has an offset on this loop index
    /// </summary>
    internal class HasOffsetIndices : ICompilerAttribute
    {
    }
}