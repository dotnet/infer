// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Compiler
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    public class ExpressionEvaluator
    {
        /// <summary>
        /// Helps build class declarations
        /// </summary>
        private static readonly CodeBuilder Builder = CodeBuilder.Instance;

        public static readonly string[] binaryOperatorNames =
            {
                "op_Addition", "op_Subtraction", "op_Multiply", "op_Division",
                "op_Modulus", "op_Shiftleft", "op_ShiftRight",
                "op_Equality", "op_Inequality", "op_Equality", "op_Inequality",
                "op_BitwiseOr", "op_BitwiseAnd", "op_BitwiseXor", "op_BooleanOr",
                "op_BooleanAnd", "op_LessThan", "op_LessThanOrEqual", "op_GreaterThan", "op_GreaterThanOrEqual"
            };

        public static readonly string[] unaryOperatorNames =
            {
                "op_UnaryNegation", "op_BooleanNot", "op_OnesComplement"
            };

        public object Evaluate(IExpression expr)
        {
            if (expr is IObjectCreateExpression) return Evaluate((IObjectCreateExpression) expr);
            else if (expr is ILiteralExpression) return ((ILiteralExpression) expr).Value;
            else if (expr is ICastExpression) return Evaluate(((ICastExpression) expr).Expression);
            else if (expr is ICheckedExpression) return Evaluate(((ICheckedExpression)expr).Expression);
            else if (expr is IBinaryExpression)
            {
                IBinaryExpression ibe = (IBinaryExpression) expr;
                object left = Evaluate(ibe.Left);
                object right = Evaluate(ibe.Right);
                Type type = left.GetType();
                return Microsoft.ML.Probabilistic.Compiler.Reflection.Invoker.InvokeStatic(type, binaryOperatorNames[(int) ibe.Operator], left, right);
            }
            else if (expr is IUnaryExpression)
            {
                IUnaryExpression iue = (IUnaryExpression) expr;
                object target = Evaluate(iue.Expression);
                Type type = target.GetType();
                return Microsoft.ML.Probabilistic.Compiler.Reflection.Invoker.InvokeStatic(type, unaryOperatorNames[(int) iue.Operator], target);
            }
            else if (expr is IMethodInvokeExpression)
            {
                IMethodInvokeExpression imie = (IMethodInvokeExpression) expr;
                object[] args = EvaluateAll(imie.Arguments);
                return Invoke(imie.Method, args);
            }
            else if (expr is IArrayCreateExpression)
            {
                IArrayCreateExpression iace = (IArrayCreateExpression) expr;
                Type t = Builder.ToType(iace.Type);
                int[] lens = new int[iace.Dimensions.Count];
                for (int i = 0; i < lens.Length; i++) lens[i] = (int) Evaluate(iace.Dimensions[i]);
                // TODO: evaluate initializer
                if (iace.Initializer != null)
                    throw new NotImplementedException("IArrayCreateExpression has an initializer block");
                return Array.CreateInstance(t, lens);
            }
            else if (expr is IFieldReferenceExpression)
            {
                IFieldReferenceExpression ifre = (IFieldReferenceExpression) expr;
                if (ifre.Target is ITypeReferenceExpression)
                {
                    ITypeReferenceExpression itre = (ITypeReferenceExpression) ifre.Target;
                    Type type = Builder.ToType(itre.Type);
                    FieldInfo info = type.GetField(ifre.Field.Name);
                    return info.GetValue(null);
                }
                else
                {
                    object target = Evaluate(ifre.Target);
                    FieldInfo info = target.GetType().GetField(ifre.Field.Name);
                    return info.GetValue(target);
                }
            }
            else throw new InferCompilerException("Could not evaluate: " + expr);
        }

        public object Invoke(IMethodReferenceExpression imre, object[] args)
        {
            object target = null;
            if (!(imre.Target is ITypeReferenceExpression)) target = Evaluate(imre.Target);
            MethodBase mb = Builder.ToMethod(imre.Method);
            return Util.Invoke(mb, target, args);
        }

        public object[] EvaluateAll(IList<IExpression> iec)
        {
            object[] objs = new object[iec.Count];
            for (int i = 0; i < objs.Length; i++) objs[i] = Evaluate(iec[i]);
            return objs;
        }

        public object Evaluate(IObjectCreateExpression ioce)
        {
            Type t = Builder.ToType(ioce.Type);
            object[] args = new object[ioce.Arguments.Count];
            for (int i = 0; i < args.Length; i++) args[i] = Evaluate(ioce.Arguments[i]);
            return Activator.CreateInstance(t, args);
        }


        /// <summary>
        /// Quotes an object instance or returns null if it cannot.  
        /// </summary>
        public static IExpression Quote(object p)
        {
            if (p is string || p.GetType().IsPrimitive || p is Enum) return Builder.LiteralExpr(p);
            else if (p is Type) return Builder.TypeRefExpr((Type) p);
            else if (p is Array) return QuoteArray((Array) p);
            else return null;
        }

        private static IExpression QuoteArray(Array array)
        {
            IArrayCreateExpression ace = Builder.ArrayCreateExpr(array.GetType().GetElementType());
            for (int i = 0; i < array.Rank; i++) ace.Dimensions.Add(Builder.LiteralExpr(array.GetLength(i)));
            ace.Initializer = Builder.BlockExpr();
            if (array.Rank == 1)
            {
                foreach (object obj in array)
                {
                    IExpression objExpr = Quote(obj);
                    if (objExpr == null) return null;
                    ace.Initializer.Expressions.Add(objExpr);
                }
                return ace;
            }
            if (array.Rank == 2)
            {
                for (int i = 0; i < array.GetLength(0); i++)
                {
                    IBlockExpression be = Builder.BlockExpr();
                    ace.Initializer.Expressions.Add(be);
                    for (int j = 0; j < array.GetLength(1); j++)
                    {
                        IExpression objExpr = Quote(array.GetValue(i, j));
                        if (objExpr == null) return null;
                        be.Expressions.Add(objExpr);
                    }
                }
                return ace;
            }
            return null;
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}