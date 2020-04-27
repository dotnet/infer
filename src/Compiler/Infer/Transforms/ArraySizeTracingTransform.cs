// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using System.Collections;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Adds statements that print the size of each message array.  During inference, these array sizes will appear on the console.
    /// </summary>
    public class ArraySizeTracingTransform : ShallowCopyTransform
    {
        Set<IVariableDeclaration> declarations = new Set<IVariableDeclaration>();
        bool isIterationLoop;

        public override string Name
        {
            get
            {
                return "ArraySizeTracingTransform";
            }
        }

        protected override IStatement ConvertFor(IForStatement ifs)
        {
            bool wasIterationLoop = isIterationLoop;
            var loopVar = Recognizer.LoopVariable(ifs);
            if (loopVar.Name == "iteration")
            {
                declarations.Clear();
                isIterationLoop = true;
                this.ShallowCopy = true;
            }
            else
                isIterationLoop = false;
            IStatement ist = base.ConvertFor(ifs);
            if (isIterationLoop)
            {
                IForStatement fs = (IForStatement)ist;
                WriteArraySizes(fs.Body.Statements);
                this.ShallowCopy = false;
            }
            isIterationLoop = wasIterationLoop;
            return ist;
        }

        private void WriteArraySizes(ICollection<IStatement> outputs)
        {
            foreach (var ivd in declarations)
            {
                WriteArraySize(outputs, ivd);
            }
            var itd = context.FindAncestor<ITypeDeclaration>();
            foreach (var ifd in itd.Fields)
            {
                WriteArraySize(outputs, ifd);
            }
        }

        private void WriteArraySize(ICollection<IStatement> outputs, object decl)
        {
            string format = "{0} {1}";
            IExpression formatExpr = Builder.LiteralExpr(format);
            VariableInformation varInfo = VariableInformation.GetVariableInformation(context, decl);
            IExpression varRefExpr;
            if (decl is IVariableDeclaration)
                varRefExpr = Builder.VarRefExpr((IVariableDeclaration)decl);
            else
                varRefExpr = Builder.FieldRefExpr((IFieldDeclaration)decl);
            var nameExpr = Builder.LiteralExpr(varRefExpr.ToString());
            var sizeExpr = GetArraySizeExpression(varInfo);
            if (sizeExpr != null)
            {
                var writeExpr = Builder.StaticMethod(new Action<string, object, object>(Console.WriteLine).Method, formatExpr, sizeExpr, nameExpr);
                var writeStmt = Builder.ExprStatement(writeExpr);
                outputs.Add(writeStmt);
            }
        }

        private IExpression GetArraySizeExpression(VariableInformation varInfo)
        {
            IExpression size = null;
            for (int bracket = varInfo.sizes.Count - 1; bracket >= 0; bracket--)
            {
                for (int i = varInfo.sizes[bracket].Length - 1; i >= 0; i--)
                {
                    if (size == null)
                        size = varInfo.sizes[bracket][i];
                    else
                    {
                        var x = Enumerable.Range(0, 4).Sum(j => 5);
                        var rangeExpr = Builder.StaticMethod(new Func<int, int, IEnumerable<int>>(Enumerable.Range), Builder.LiteralExpr(0), varInfo.sizes[bracket][i]);
                        var delegateExpr = Builder.AnonMethodExpr(typeof(Func<int, int>));
                        delegateExpr.Parameters.Add(Builder.Param(varInfo.indexVars[bracket][i].Name, typeof(int)));
                        delegateExpr.Body = Builder.BlockStmt();
                        delegateExpr.Body.Statements.Add(Builder.Return(size));
                        var sumExpr = Builder.Method(rangeExpr, new Func<IEnumerable<int>, Func<int, int>, int>(System.Linq.Enumerable.Sum<int>), delegateExpr);
                        size = sumExpr;
                    }
                }
            }
            if (size == null && typeof(IEnumerable).IsAssignableFrom(varInfo.varType))
                size = Builder.StaticMethod(new Func<IEnumerable, long>(JaggedArray.GetLongLength), varInfo.GetExpression());
            return size;
        }

        protected override IVariableDeclaration ConvertVariableDecl(IVariableDeclaration ivd)
        {
            if (isIterationLoop)
            {
                declarations.Add(ivd);
            }
            return ivd;
        }
    }
}
