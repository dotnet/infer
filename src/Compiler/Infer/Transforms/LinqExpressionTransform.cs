// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete;
using System.Reflection;
using Microsoft.ML.Probabilistic.Compiler;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Converts expressions in the form of a Linq Expression tree into
    /// the equivalent Infer.NET expression.
    /// </summary>
    internal class LinqExpressionTransform
    {
        private static readonly CodeBuilder Builder = CodeBuilder.Instance;

        /// <summary>
        /// Converts a LINQ expression into the Infer.NET equivalent.
        /// </summary>
        /// <param name="expression"></param>
        /// <returns></returns>
        internal IExpression Convert(System.Linq.Expressions.Expression expression)
        {
            if (expression is System.Linq.Expressions.LambdaExpression)
            {
                return ConvertLambda((System.Linq.Expressions.LambdaExpression) expression);
            }
            if (expression is System.Linq.Expressions.BinaryExpression)
            {
                return ConvertBinary((System.Linq.Expressions.BinaryExpression) expression);
            }
            if (expression is System.Linq.Expressions.MemberExpression)
            {
                return ConvertMember((System.Linq.Expressions.MemberExpression) expression);
            }
            if (expression is System.Linq.Expressions.ParameterExpression)
            {
                return ConvertParameterRef((System.Linq.Expressions.ParameterExpression) expression);
            }
            if (expression is System.Linq.Expressions.ConstantExpression)
            {
                return ConvertConstant((System.Linq.Expressions.ConstantExpression) expression);
            }
            if (expression is System.Linq.Expressions.MethodCallExpression)
            {
                return ConvertMethodCall((System.Linq.Expressions.MethodCallExpression) expression);
            }

            throw new NotImplementedException("Could not convert expression of type " + expression.GetType().Name + ": " + expression);
        }

        /// <summary>
        /// Converts a LINQ method call expression into the Infer.NET equivalent.
        /// </summary>
        /// <param name="methodCallExpression"></param>
        /// <returns></returns>
        private IExpression ConvertMethodCall(System.Linq.Expressions.MethodCallExpression methodCallExpression)
        {
            var args = methodCallExpression.Arguments.Select(arg => Convert(arg)).ToArray();
            var me = Builder.Method(Convert(methodCallExpression.Object), methodCallExpression.Method, args);
            return me;
        }

        /// <summary>
        /// Converts a LINQ constant expression into the Infer.NET equivalent.
        /// </summary>
        /// <param name="constantExpression"></param>
        /// <returns></returns>
        private IExpression ConvertConstant(System.Linq.Expressions.ConstantExpression constantExpression)
        {
            return Builder.LiteralExpr(constantExpression.Value);
        }

        /// <summary>
        /// Converts a LINQ parameter expression into the Infer.NET equivalent.
        /// </summary>
        /// <param name="parameterExpression"></param>
        /// <returns></returns>
        private IExpression ConvertParameterRef(System.Linq.Expressions.ParameterExpression parameterExpression)
        {
            return Builder.ParamRef(ConvertParameter(parameterExpression));
        }

        /// <summary>
        /// Converts a LINQ member expression into the Infer.NET equivalent.
        /// </summary>
        /// <param name="memberExpression"></param>
        /// <returns></returns>
        private IExpression ConvertMember(System.Linq.Expressions.MemberExpression memberExpression)
        {
            if (memberExpression.Member is PropertyInfo)
            {
                var pi = (PropertyInfo) memberExpression.Member;
                return Builder.PropRefExpr(Convert(memberExpression.Expression), pi.DeclaringType, pi.Name);
            }
            throw new NotImplementedException("Could not convert member expression of type " + memberExpression.GetType().Name + ": " + memberExpression);
        }

        /// <summary>
        /// Converts a LINQ binary expression into the Infer.NET equivalent.
        /// </summary>
        /// <param name="binaryExpression"></param>
        /// <returns></returns>
        private IExpression ConvertBinary(System.Linq.Expressions.BinaryExpression binaryExpression)
        {
            return Builder.BinaryExpr(Convert(binaryExpression.Left),
                                      ConvertBinaryOp(binaryExpression.NodeType),
                                      Convert(binaryExpression.Right)
                );
        }

        /// <summary>
        /// Converts a LINQ binary operator into the Infer.NET equivalent.
        /// </summary>
        /// <param name="nodeType"></param>
        /// <returns></returns>
        private BinaryOperator ConvertBinaryOp(System.Linq.Expressions.ExpressionType nodeType)
        {
            if (nodeType == System.Linq.Expressions.ExpressionType.Add) return BinaryOperator.Add;
            if (nodeType == System.Linq.Expressions.ExpressionType.And) return BinaryOperator.BitwiseAnd;
            if (nodeType == System.Linq.Expressions.ExpressionType.ExclusiveOr) return BinaryOperator.BitwiseExclusiveOr;
            if (nodeType == System.Linq.Expressions.ExpressionType.Or) return BinaryOperator.BitwiseOr;
            if (nodeType == System.Linq.Expressions.ExpressionType.AndAlso) return BinaryOperator.BooleanAnd;
            if (nodeType == System.Linq.Expressions.ExpressionType.OrElse) return BinaryOperator.BooleanOr;
            if (nodeType == System.Linq.Expressions.ExpressionType.Divide) return BinaryOperator.Divide;
            if (nodeType == System.Linq.Expressions.ExpressionType.GreaterThan) return BinaryOperator.GreaterThan;
            if (nodeType == System.Linq.Expressions.ExpressionType.GreaterThanOrEqual) return BinaryOperator.GreaterThanOrEqual;
            if (nodeType == System.Linq.Expressions.ExpressionType.Equal) return BinaryOperator.ValueEquality;
            if (nodeType == System.Linq.Expressions.ExpressionType.NotEqual) return BinaryOperator.ValueInequality;
            if (nodeType == System.Linq.Expressions.ExpressionType.LessThan) return BinaryOperator.LessThan;
            if (nodeType == System.Linq.Expressions.ExpressionType.LessThanOrEqual) return BinaryOperator.LessThanOrEqual;
            if (nodeType == System.Linq.Expressions.ExpressionType.Modulo) return BinaryOperator.Modulus;
            if (nodeType == System.Linq.Expressions.ExpressionType.Multiply) return BinaryOperator.Multiply;
            if (nodeType == System.Linq.Expressions.ExpressionType.LeftShift) return BinaryOperator.ShiftLeft;
            if (nodeType == System.Linq.Expressions.ExpressionType.RightShift) return BinaryOperator.ShiftRight;
            if (nodeType == System.Linq.Expressions.ExpressionType.Subtract) return BinaryOperator.Subtract;
            throw new NotImplementedException("Could not convert operator: " + nodeType);
        }

        /// <summary>
        /// Converts a LINQ lambda expression into the Infer.NET equivalent.
        /// </summary>
        /// <param name="lambdaExpression"></param>
        /// <returns></returns>
        internal ILambdaExpression ConvertLambda(System.Linq.Expressions.LambdaExpression lambdaExpression)
        {
            var lambda = new XLambdaExpression
                {
                    Body = Convert(lambdaExpression.Body),
                };
            foreach (var p in lambdaExpression.Parameters)
            {
                lambda.Parameters.Add(ConvertParameterToVar(p));
            }
            return lambda;
        }

        /// <summary>
        /// Converts a parameter expression into the Infer.NET equivalent variable declaration.
        /// (needed because we use variable declarations instead of parameter declarations for lambda parameters).
        /// </summary>
        /// <param name="p"></param>
        /// <returns></returns>
        internal IVariableDeclaration ConvertParameterToVar(System.Linq.Expressions.ParameterExpression p)
        {
            // todo: should the lambda expression be using variable declarations rather than
            // parameter declarations?
            return Builder.VarDecl(p.Name, p.Type);
        }

        /// <summary>
        /// Converts a parameter expression into the Infer.NET equivalent
        /// </summary>
        /// <param name="p"></param>
        /// <returns></returns>
        internal IParameterDeclaration ConvertParameter(System.Linq.Expressions.ParameterExpression p)
        {
            return Builder.Param(p.Name, p.Type);
        }
    }
}