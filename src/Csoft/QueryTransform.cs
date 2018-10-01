// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using System.Collections.Generic;
using System.Linq.Expressions;
using System.Reflection;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Factors;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    public class QueryTransform : ShallowCopyTransform
    {
        private InferenceEngine engine;
        private MethodInfo singleQuery;

        // cache of analysis results
        private List<IMemberReference> observedVariables = new List<IMemberReference>();
        private List<IMemberReference> inferredVariables = new List<IMemberReference>();

        private IVariableDeclaration algorithmVar;
        private List<ITypeDeclaration> algorithmTypeDecls = new List<ITypeDeclaration>(); 
        private ITypeDeclaration originalTypeDecl;

        public QueryTransform(InferenceEngine engine, MethodInfo singleQuery)
        {
            this.engine = engine;
            this.singleQuery = singleQuery;
        }

        public override ITypeDeclaration ConvertType(ITypeDeclaration itd)
        {
            originalTypeDecl = itd;
            var td = base.ConvertType(itd);
            td.Name = itd.Name + "Generated";
            if (singleQuery == null)
            {
                td.BaseType = Builder.TypeRef(itd.Name, itd.DotNetType, itd, itd.GenericArguments.ToArray());
            }
            else
            {
                var t = typeof (ISingleQuery);
                td.BaseType = Builder.TypeRef(t.Name, t, null, new IType[0]);
            }
            td.NestedTypes.AddRange(algorithmTypeDecls);
            return td;
        }

        protected override IMethodDeclaration ConvertMethod(IMethodDeclaration imd)
        {
            if (singleQuery == null && !imd.Virtual)
            {
                return null;
            }
            if (singleQuery != null && imd.MethodInfo != singleQuery)
            {
                return null;
            }

            // TODO check method contains at least one infer call

            var analysisTransform = new QueryAnalysisTransform();
            analysisTransform.ScanMethod(imd);
            inferredVariables = analysisTransform.toInfer;
            observedVariables = analysisTransform.toObserve;

            var converted = base.ConvertMethod(imd);
            if (singleQuery == null)
            {
                converted.Overrides = true;
            }
            else
            {
                converted.Static = false;
                converted.Visibility = MethodVisibility.Public;
                converted.Name = GetMethodInfo<ISingleQuery>(sq => sq.ExecuteQuery()).Name;
            }
            return converted;
        }

        protected override IPropertyDeclaration ConvertProperty(ITypeDeclaration td, IPropertyDeclaration ipd, bool convertGetterAndSetter = true)
        {
            return null;
        }

        protected override IFieldDeclaration ConvertField(ITypeDeclaration td, IFieldDeclaration ifd)
        {
            return null;
        }

        protected override IStatement ConvertExpressionStatement(IExpressionStatement ies)
        {
            var imie = ies.Expression as IMethodInvokeExpression;
            if (imie == null)
            {
                return base.ConvertExpressionStatement(ies);
            }

            if (CodeRecognizer.Instance.IsStaticGenericMethod(imie, new Action<object, object>(Csoft.Observe)))
            {
                var memberRef = ExtractMemberRef(imie.Arguments[0]);
                if (memberRef == null)
                {
                    throw new InvalidOperationException("First argument must be a field or property");
                }

                var observedExpr = imie.Arguments[1];

                var observeMethod = GetMethodInfo<IGeneratedAlgorithm>(algo => algo.SetObservedValue(null, null));
                var call = Builder.Method(Builder.VarRefExpr(algorithmVar), observeMethod, Builder.LiteralExpr(memberRef.Name), observedExpr);
                return Builder.ExprStatement(call);
            }

            if (Attribute.GetCustomAttribute(imie.Method.Method.MethodInfo, typeof(ModelMethodAttribute)) != null)
            {
                return ConvertModelMethodCall(imie);
            }

            return base.ConvertExpressionStatement(ies);
        }

        private IStatement ConvertModelMethodCall(IMethodInvokeExpression imie)
        {
            var modelMethodDecl = imie.Method.Method.Resolve();
            var modelMethodParamDecls = modelMethodDecl.Parameters.ToArray();

            foreach (var var in inferredVariables)
            {
                var inferMethod = new Action<object, string>(InferNet.Infer);
                var call = Builder.StaticMethod(inferMethod, BuildMemberRefExpr(var), Builder.LiteralExpr(var.Name));
                modelMethodDecl.Body.Statements.Add(Builder.ExprStatement(call));
            }

            foreach (var var in observedVariables)
            {
                modelMethodDecl.Parameters.Add(Builder.Param(var.Name, GetMemberReferenceType(var)));
            }

            var tds = engine.Compiler.GetTransformedDeclaration(originalTypeDecl, imie.Method.Method.MethodInfo, new AttributeRegistry<object, ICompilerAttribute>(true));
            var algorithmTypeDecl = tds.Single();

            // give algorithm a unique name
            algorithmTypeDecl.Name = algorithmTypeDecl.Name + (algorithmTypeDecls.Count + 1);
            algorithmTypeDecls.Add(algorithmTypeDecl);

            // TODO fix shortcomings in the code model
            var newObj = Builder.ObjCreateExpr();
            newObj.Type = algorithmTypeDecl;

            algorithmVar = Builder.VarDecl("algorithm", algorithmTypeDecl);

            // handle model method parameters
            for(int i=0; i<imie.Arguments.Count; i++)
            {
                var paramExpr = imie.Arguments[i];
                var paramDecl = modelMethodParamDecls[i];
                var observeMethod = GetMethodInfo<IGeneratedAlgorithm>(algo => algo.SetObservedValue(null, null));
                var call = Builder.Method(Builder.VarRefExpr(algorithmVar), observeMethod, Builder.LiteralExpr(paramDecl.Name), paramExpr);
                context.AddStatementAfterCurrent(Builder.ExprStatement(call));
            }

            return Builder.AssignStmt(Builder.VarDeclExpr(algorithmVar), newObj);
        }

        protected override IExpression ConvertMethodInvoke(IMethodInvokeExpression imie)
        {
            if (CodeRecognizer.Instance.IsStaticGenericMethod(imie, new Func<PlaceHolder, object>(Csoft.Infer<PlaceHolder>)))
            {
                var memberRef = ExtractMemberRef(imie.Arguments[0]);
                if (memberRef == null)
                {
                    throw new InvalidOperationException("First argument must be a field or property");
                }

                var executeMethod = GetMethodInfo<IGeneratedAlgorithm>(algo => algo.Execute(0));
                var executeCall = Builder.Method(Builder.VarRefExpr(algorithmVar), executeMethod, Builder.LiteralExpr(engine.NumberOfIterations));
                context.AddStatementBeforeCurrent(Builder.ExprStatement(executeCall));

                var marginalMethod = GetMethodInfo<IGeneratedAlgorithm>(algo => algo.Marginal<Bernoulli>(null)).GetGenericMethodDefinition();
                var specificMarginalMethod = marginalMethod.MakeGenericMethod(imie.Method.Method.MethodInfo.GetGenericArguments()[0]);
                return Builder.Method(Builder.VarRefExpr(algorithmVar), specificMarginalMethod, Builder.LiteralExpr(memberRef.Name));
            }

            return base.ConvertMethodInvoke(imie);
        }

        private static IType GetMemberReferenceType(IMemberReference var)
        {
            IType varType;
            if (var is IPropertyReference)
            {
                varType = ((IPropertyReference)var).PropertyType;
            }
            else if (var is IFieldReference)
            {
                varType = ((IFieldReference)var).FieldType;
            }
            else
            {
                throw new InvalidOperationException();
            }
            return varType;
        }

        public static MethodInfo GetMethodInfo<T>(Expression<Action<T>> expression)
        {
            var member = expression.Body as MethodCallExpression;
            if (member != null)
            {
                return member.Method;
            }
            throw new InvalidOperationException();
        }

        private static IMemberReference ExtractMemberRef(IExpression expr)
        {
            var propRefExpr = expr as IPropertyReferenceExpression;
            if (propRefExpr != null)
            {
                return propRefExpr.Property;
            }
            var fieldRefExpr = expr as IFieldReferenceExpression;
            return fieldRefExpr != null ? fieldRefExpr.Field : null;
        }

        private static IExpression BuildMemberRefExpr(IMemberReference mr)
        {
            var propRef = mr as IPropertyReference;
            if (propRef != null)
            {
                return Builder.PropRefExpr(Builder.ThisRefExpr(), propRef);
            }

            var fieldRef = mr as IFieldReference;
            if (fieldRef != null)
            {
                return Builder.FieldRefExpr(fieldRef);
            }

            throw new InvalidOperationException();
        }

        private class QueryAnalysisTransform : ShallowCopyTransform
        {
            public List<IMemberReference> toInfer = new List<IMemberReference>();
            public List<IMemberReference> toObserve = new List<IMemberReference>();

            public void ScanMethod(IMethodDeclaration method)
            {
                DoConvertMethodBody(Builder.BlockStmt().Statements, method.Body.Statements);
            }

            protected override IExpression ConvertMethodInvoke(IMethodInvokeExpression imie)
            {
                if (CodeRecognizer.Instance.IsStaticGenericMethod(imie, new Func<PlaceHolder, object>(Csoft.Infer<PlaceHolder>)))
                {
                    var memberRef = ExtractMemberRef(imie.Arguments[0]);
                    if (memberRef != null)
                    {
                        toInfer.Add(memberRef);
                    }
                }
                if (CodeRecognizer.Instance.IsStaticGenericMethod(imie, new Action<object, object>(Csoft.Observe)))
                {
                    var memberRef = ExtractMemberRef(imie.Arguments[0]);
                    if (memberRef != null)
                    {
                        toObserve.Add(memberRef);
                    }
                }
                return base.ConvertMethodInvoke(imie);
            }
        }
    }

    public interface ISingleQuery
    {
        void ExecuteQuery();
    }
}
