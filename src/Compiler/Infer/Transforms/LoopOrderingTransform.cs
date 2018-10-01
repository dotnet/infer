// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Order loops with literal sizes by variable name.  Push conditionals inside of sequential loops.
    /// </summary>
    internal class LoopOrderingTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "LoopOrderingTransform"; }
        }

        protected List<IVariableDeclaration> sequentialLoops = new List<IVariableDeclaration>();
        private bool containsLoopVar;

        protected override IStatement ConvertCondition(IConditionStatement ics)
        {
            ics = (IConditionStatement) base.ConvertCondition(ics);
            if (ics != null && ics.Then.Statements.Count == 1)
            {
                IStatement st = ics.Then.Statements[0];
                if (st is IForStatement)
                {
                    IForStatement ifs = (IForStatement) st;
                    IVariableDeclaration loopVar = Recognizer.LoopVariable(ifs);
                    bool isSequential = context.InputAttributes.Has<Sequential>(loopVar);
                    if (isSequential)
                    {
                        ics.Then = ifs.Body;
                        ifs.Body = Builder.BlockStmt();
                        ifs.Body.Statements.Add(ics);
                        return ifs;
                    }
                }
            }
            return ics;
        }

        protected override IExpression ConvertVariableRefExpr(IVariableReferenceExpression ivre)
        {
            // TODO: make this more precise
            containsLoopVar = true;
            return base.ConvertVariableRefExpr(ivre);
        }

        protected override IStatement ConvertFor(IForStatement ifs)
        {
            if (context.InputAttributes.Has<ConvergenceLoop>(ifs))
            {
                SerialSchedulingInfo info = context.InputAttributes.Get<SerialSchedulingInfo>(ifs);
                if (info != null)
                {
                    sequentialLoops.Clear();
                    foreach (SerialLoopInfo loopInfo in info.loopInfos)
                    {
                        sequentialLoops.Add(loopInfo.loopVar);
                    }
                }
            }
            ifs = (IForStatement) base.ConvertFor(ifs);
            // collect a list of all nested loops
            List<IForStatement> loops = new List<IForStatement>();
            List<IForStatement> otherLoops = new List<IForStatement>();
            IForStatement fs = ifs;
            while (true)
            {
                IExpression loopStart = Recognizer.LoopStartExpression(fs);
                IExpression loopSize = Recognizer.LoopSizeExpression(fs);
                containsLoopVar = false;
                ConvertExpression(loopStart);
                ConvertExpression(loopSize);
                if (!containsLoopVar) loops.Add(fs);
                else otherLoops.Add(fs);
                if (fs is IBrokenForStatement) break;
                if (fs.Body.Statements.Count != 1) break;
                if (!(fs.Body.Statements[0] is IForStatement)) break;
                fs = (IForStatement) fs.Body.Statements[0];
            }
            if (loops.Count + otherLoops.Count == 1) return ifs;
            IList<IStatement> innermostBody = fs.Body.Statements;
            innermostBody = Containers.WrapWithContainers(innermostBody, otherLoops.ConvertAll(s => (IStatement)s));
            loops.Sort(CompareLoops);
            IStatement wrapped = Containers.WrapWithContainers(innermostBody, loops.ConvertAll(s => (IStatement)s))[0];
            context.InputAttributes.CopyObjectAttributesTo(ifs, context.OutputAttributes, wrapped);
            return wrapped;
        }

        public int CompareLoops(IForStatement ifs1, IForStatement ifs2)
        {
            IVariableDeclaration loopVar1 = Recognizer.LoopVariable(ifs1);
            //IExpression loopSize1 = Recognizer.LoopSizeExpression(ifs1);
            //if (!(loopSize1 is ILiteralExpression)) throw new InferCompilerException("loop size is not literal");
            IVariableDeclaration loopVar2 = Recognizer.LoopVariable(ifs2);
            //IExpression loopSize2 = Recognizer.LoopSizeExpression(ifs2);
            //if (!(loopSize2 is ILiteralExpression)) throw new InferCompilerException("loop size is not literal");
            int index1 = sequentialLoops.IndexOf(loopVar1);
            int index2 = sequentialLoops.IndexOf(loopVar2);
            if (index1 == -1)
            {
                if (index2 == -1)
                {
                    return String.Compare(loopVar1.Name, loopVar2.Name, StringComparison.InvariantCulture);
                }
                else return 1; // loopVar2 wins
            }
            else if (index2 == -1) return -1; // loopVar1 wins
            else return index1.CompareTo(index2); // lower index goes first
        }
    }
}