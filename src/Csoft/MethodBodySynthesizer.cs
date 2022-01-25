// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler
{
    public class MethodBodySynthesizer
    {
        private static readonly CodeBuilder Builder = CodeBuilder.Instance;

        private readonly SemanticModel model;
        private readonly IType declaringType;
        private readonly Dictionary<string, ITypeDeclaration> typeDecls;

        private MethodBodySynthesizer(SemanticModel model, IType declaringType, Dictionary<string, ITypeDeclaration> typeDecls)
        {
            this.model = model;
            this.declaringType = declaringType;
            this.typeDecls = typeDecls;
        }

        public static void SynthesizeBody(IMethodDeclaration methodDecl, MethodDeclarationSyntax methodSyntax, SemanticModel model, Dictionary<string, ITypeDeclaration> typeDecls)
        {
            var synthesizer = new MethodBodySynthesizer(model, methodDecl.DeclaringType, typeDecls);
            synthesizer.SynthesizeBody(methodDecl, methodSyntax);
        }

        private void SynthesizeBody(IMethodDeclaration methodDecl, MethodDeclarationSyntax methodSyntax)
        {
            pdCache.Clear();
            vdCache.Clear();
            foreach (var paramDecl in methodDecl.Parameters)
            {
                pdCache[paramDecl.Name] = paramDecl;
            }
            if (methodSyntax.Body != null)
                methodDecl.Body = ConvertBlock(methodSyntax.Body);
            pdCache.Clear();
            vdCache.Clear();
        }

        private IType ConvertTypeReference(ITypeSymbol typeSymbol)
        {
            return TypeSymbolConverter.ConvertTypeReference(typeSymbol, model.Compilation.Assembly, declaringType.DotNetType.Assembly);
        }

        private Type ConvertTypeSymbolToType(ITypeSymbol typeSymbol)
        {
            return TypeSymbolConverter.ConvertTypeSymbolToType(typeSymbol, model.Compilation.Assembly, declaringType.DotNetType.Assembly);
        }

        private IBlockStatement ConvertBlock(BlockSyntax block)
        {
            var bs = Builder.BlockStmt();
            foreach (var st in block.Statements)
            {
                var st2 = ConvertStatement(st);
                if (st2 != null) bs.Statements.Add(st2);
            }
            return bs;
        }

        private IBlockStatement ConvertBlock(StatementSyntax statement)
        {
            if (statement is BlockSyntax blockSyntax)
            {
                return ConvertBlock(blockSyntax);
            }
            var bs = Builder.BlockStmt();
            var st2 = ConvertStatement(statement);
            bs.Statements.Add(st2);
            return bs;
        }

        private IStatement ConvertStatement(StatementSyntax statement)
        {
            if (statement is LocalDeclarationStatementSyntax ldss)
            {
                return ConvertLocalDeclarationStatement(ldss);
            }
            if (statement is ExpressionStatementSyntax ess)
            {
                return ConvertExpressionStatement(ess);
            }
            if (statement is IfStatementSyntax ifss)
            {
                return ConvertIfStatement(ifss);
            }
            if (statement is ForStatementSyntax fss)
            {
                return ConvertForStatement(fss);
            }
            if (statement is BlockSyntax bs)
            {
                return ConvertBlock(bs);
            }
            if (statement is UsingStatementSyntax uss)
            {
                return ConvertUsingStatement(uss);
            }
            if (statement is EmptyStatementSyntax)
            {
                return new XBlockStatement();
            }
            if (statement is TryStatementSyntax)
            {
                throw new NotSupportedException("Unsupported statement type: " + statement);
            }
            if (statement is ThrowStatementSyntax tts)
            {
                return Builder.ThrowStmt(ConvertExpression(tts.Expression));
            }
            if (statement is ReturnStatementSyntax rss)
            {
                return ConvertReturnStatement(rss);
            }
            throw new NotSupportedException("Unsupported statement type: " + statement);
        }

        private IMethodReturnStatement ConvertReturnStatement(ReturnStatementSyntax rss)
        {
            var mrs = Builder.MethodRtrnStmt();
            mrs.Expression = ConvertExpression(rss.Expression);
            return mrs;
        }

        private IStatement ConvertUsingStatement(UsingStatementSyntax usingStatement)
        {
            return new XUsingStatement
            {
                Expression = usingStatement.Declaration != null ?
                ConvertVariableDeclaration(usingStatement.Declaration) :
                ConvertExpression(usingStatement.Expression),
                Body = ConvertBlock(usingStatement.Statement)
            };
        }

        private IStatement ConvertForStatement(ForStatementSyntax forStatement)
        {
            var variableDecl = ConvertVariableDeclaration(forStatement.Declaration);
            var incrementors = forStatement.Incrementors.Select(i => ConvertExpression(i)).ToList();
            var initializers = forStatement.Initializers.Select(i => ConvertExpression(i)).ToList();
            return new XForStatement
            {
                Body = ConvertBlock(forStatement.Statement),
                Condition = ConvertExpression(forStatement.Condition),
                Increment = new XExpressionStatement { Expression = incrementors.Single() },
                Initializer = new XExpressionStatement { Expression = variableDecl }
            };
        }

        private IStatement ConvertIfStatement(IfStatementSyntax ifStatement)
        {
            var condition = ConvertExpression(ifStatement.Condition);
            var statement = ConvertBlock(ifStatement.Statement);
            if (ifStatement.Else == null || ifStatement.Else.Statement == null)
            {
                return Builder.CondStmt(condition, statement);
            }
            var elseStmt = ConvertBlock(ifStatement.Else.Statement);
            return Builder.CondStmt(condition, statement, elseStmt);
        }

        private IStatement ConvertExpressionStatement(ExpressionStatementSyntax expressionStatement)
        {
            var expr = ConvertExpression(expressionStatement.Expression);
            return Builder.ExprStatement(expr);
        }

        private IStatement ConvertLocalDeclarationStatement(LocalDeclarationStatementSyntax localDeclarationStatement)
        {
            var es = Builder.ExprStatement();
            es.Expression = ConvertVariableDeclaration(localDeclarationStatement.Declaration);
            return es;
        }

        private IExpression ConvertVariableDeclaration(VariableDeclarationSyntax variableDeclaration)
        {
            if (variableDeclaration.Variables.Count != 1) throw new NotSupportedException("Multiple variable declarations not supported");
            var variableDeclarator = variableDeclaration.Variables.First();
            var vd = Builder.VarDecl();
            vd.Name = variableDeclarator.Identifier.ValueText;
            vd.VariableType = ConvertTypeReference(model.GetTypeInfo(variableDeclaration.Type).Type);
            vdCache[vd.Name] = vd;
            var vde = Builder.VarDeclExpr(vd);
            if (variableDeclarator.Initializer != null)
            {
                return Builder.AssignExpr(vde, ConvertExpression(variableDeclarator.Initializer.Value));
            }
            return vde;
        }

        private IExpression ConvertExpression(ExpressionSyntax expression)
        {
            if (expression is InvocationExpressionSyntax ies)
            {
                return ConvertInvocation(ies);
            }
            if (expression is LiteralExpressionSyntax literalExpressionSyntax)
            {
                return ConvertLiteral(literalExpressionSyntax);
            }
            if (expression is MemberAccessExpressionSyntax maes)
            {
                return ConvertMemberAccess(maes);
            }
            if (expression is IdentifierNameSyntax ins)
            {
                return ConvertIdentifierName(ins);
            }
            if (expression is BinaryExpressionSyntax bes)
            {
                return ConvertBinary(bes);
            }
            if (expression is ObjectCreationExpressionSyntax oces)
            {
                return ConvertCreation(oces);
            }
            if (expression is ElementAccessExpressionSyntax eaes)
            {
                return ConvertElementAccess(eaes);
            }
            if (expression is ArrayCreationExpressionSyntax aces)
            {
                return ConvertArrayCreation(aces);
            }
            if (expression is ParenthesizedExpressionSyntax pes)
            {
                return ConvertExpression(pes.Expression);
            }
            if (expression is GenericNameSyntax genericNameSyntax)
            {
                var typeSymbol = model.GetTypeInfo(genericNameSyntax).Type;
                return new XTypeReferenceExpression
                {
                    // TODO why is the horrible cast required?
                    Type = (ITypeReference)ConvertTypeReference(typeSymbol)
                };
            }
            if (expression is ConditionalExpressionSyntax ces)
            {
                return ConvertConditionalExpression(ces);
            }
            if (expression is CastExpressionSyntax castExpressionSyntax)
            {
                var typeSymbol = model.GetTypeInfo(castExpressionSyntax).Type;
                return new XCastExpression
                {
                    Expression = ConvertExpression(castExpressionSyntax.Expression),
                    TargetType = ConvertTypeReference(typeSymbol)
                };
            }
            if (expression is PostfixUnaryExpressionSyntax postfix)
            {
                UnaryOperator op;
                switch (postfix.Kind())
                {
                    // TODO complete this list and factor out
                    case SyntaxKind.PostIncrementExpression:
                        op = UnaryOperator.PostIncrement;
                        break;
                    case SyntaxKind.PostDecrementExpression:
                        op = UnaryOperator.PostDecrement;
                        break;
                    default:
                        throw new NotSupportedException("Operator " + postfix.OperatorToken.RawKind);
                }
                return new XUnaryExpression
                {
                    Expression = ConvertExpression(postfix.Operand),
                    Operator = op
                };
            }
            if (expression is PrefixUnaryExpressionSyntax prefix)
            {
                var ixe = new XUnaryExpression();
                ixe.Expression = ConvertExpression(prefix.Operand);
                switch (prefix.Kind())
                {
                    // TODO complete this list and factor out
                    case SyntaxKind.LogicalNotExpression:
                        ixe.Operator = UnaryOperator.BooleanNot;
                        break;
                    case SyntaxKind.UnaryMinusExpression:
                        if (prefix.Operand is LiteralExpressionSyntax les)
                        {
                            if (les.Token.IsKind(SyntaxKind.NumericLiteralToken))
                            {
                                var typeInfo = model.GetTypeInfo(prefix);
                                var val = les.Token.Value;
                                var newType = ConvertTypeReference(typeInfo.ConvertedType).DotNetType;
                                val = Convert.ChangeType(val, newType);
                                ixe.Expression = Builder.LiteralExpr(val);
                            }
                        }
                        ixe.Operator = UnaryOperator.Negate;
                        break;
                    default:
                        throw new NotSupportedException("Operator " + prefix.Kind());
                }
                return ixe;
            }
            if (expression is InitializerExpressionSyntax init)
            {
                var exprs = init.Expressions.Select(expr => ConvertExpression(expr)).ToList();
                var block = new XBlockExpression();
                block.Expressions.AddRange(exprs);
                return block;
            }
            if (expression is TypeOfExpressionSyntax toes)
            {
                var expr = Builder.TypeOfExpr();
                var typeSymbol = model.GetTypeInfo(toes.Type).Type;
                expr.Type = ConvertTypeReference(typeSymbol);
                return expr;
            }
            if (expression is TypeSyntax)
            {
                var typeSymbol = model.GetTypeInfo(expression).Type;
                return Builder.TypeRefExpr(ConvertTypeReference(typeSymbol));
            }
            if (expression is AssignmentExpressionSyntax assgnExpr)
            {
                var expr = Builder.AssignExpr();
                expr.Target = ConvertExpression(assgnExpr.Left);
                expr.Expression = ConvertExpression(assgnExpr.Right);
                return expr;
            }
            if (expression is CheckedExpressionSyntax checkedExpr)
            {
                var expr = Builder.CheckedExpr();
                expr.Expression = ConvertExpression(checkedExpr.Expression);
                return expr;
            }
            throw new NotSupportedException("Unsupported expression type: " + expression);
        }

        private IExpression ConvertConditionalExpression(ConditionalExpressionSyntax conditionalExpression)
        {
            var ixe = new XConditionExpression();
            ixe.Condition = ConvertExpression(conditionalExpression.Condition);
            ixe.Else = ConvertExpression(conditionalExpression.WhenFalse);
            ixe.Then = ConvertExpression(conditionalExpression.WhenTrue);
            return ixe;
        }

        private IExpression ConvertArrayCreation(ArrayCreationExpressionSyntax creationExpression)
        {
            var ixe = new XArrayCreateExpression();
            var arrayType = (IArrayTypeSymbol)model.GetTypeInfo(creationExpression).Type;
            ixe.Type = ConvertTypeReference(arrayType.ElementType);

            if (creationExpression.Initializer == null)
            {
                foreach (var size in creationExpression.Type.RankSpecifiers.First().Sizes)
                {
                    ixe.Dimensions.Add(ConvertExpression(size));
                }
                return ixe;
            }

            var arrayInitSyntax = creationExpression.Initializer;
            while (arrayInitSyntax != null)
            {
                ixe.Dimensions.Add(new XLiteralExpression { Value = arrayInitSyntax.Expressions.Count });
                if (arrayInitSyntax.Expressions.Count == 0) break;
                var firstExpr = arrayInitSyntax.Expressions.First();
                arrayInitSyntax = firstExpr as InitializerExpressionSyntax;
            }
            var initializers = creationExpression.Initializer.Expressions.Select(expr => ConvertExpression(expr)).ToList();
            ixe.Initializer = new XBlockExpression();
            ixe.Initializer.Expressions.AddRange(initializers);
            return ixe;
        }

        private IExpression ConvertElementAccess(ElementAccessExpressionSyntax accessExpression)
        {
            var ixe = new XArrayIndexerExpression();
            foreach (var indx in accessExpression.ArgumentList.Arguments)
            {
                ixe.Indices.Add(ConvertExpression(indx.Expression));
            }
            ixe.Target = ConvertExpression(accessExpression.Expression);
            return ixe;
        }

        private IExpression ConvertCreation(ObjectCreationExpressionSyntax creationExpression)
        {
            var symbol = (IMethodSymbol)model.GetSymbolInfo(creationExpression).Symbol;
            if (symbol == null)
            {
                throw new Exception("Could not bind");
            }

            var type = ConvertTypeReference(symbol.ContainingType);
            var args = creationExpression.ArgumentList != null ?
                creationExpression.ArgumentList.Arguments.Select(arg => ConvertExpression(arg.Expression)).ToArray() :
                null;

            var objCreateExpr = Builder.ObjCreateExpr();
            objCreateExpr.Type = type;
            if (args != null)
            {
                objCreateExpr.Arguments.AddRange(args);
            }

            // TODO probably need to worry about param arrays and emit explicit array creation here
            var paramTypes = symbol.Parameters.Select(p => ConvertTypeSymbolToType(p.Type)).ToArray();
            var dotNetType = ConvertTypeSymbolToType(symbol.ContainingType);
            objCreateExpr.Constructor = Builder.ConstructorRef(dotNetType, paramTypes);

            return objCreateExpr;
        }

        private IExpression ConvertBinary(BinaryExpressionSyntax binaryExpression)
        {
            var left = ConvertExpression(binaryExpression.Left);
            var right = ConvertExpression(binaryExpression.Right);
            switch (binaryExpression.Kind())
            {
                case SyntaxKind.SimpleAssignmentExpression:
                    return Builder.AssignExpr(left, right);
                case SyntaxKind.AddAssignmentExpression:
                    return Builder.AssignExpr(left, Builder.BinaryExpr(BinaryOperator.Add, left, right));
                default:
                    return Builder.BinaryExpr(left, ConvertOperator(binaryExpression.Kind()), right);
            }
        }

        private BinaryOperator ConvertOperator(SyntaxKind kind)
        {
            switch (kind)
            {
                case SyntaxKind.EqualsExpression:
                    // TODO value or identity equality?
                    return BinaryOperator.ValueEquality;
                case SyntaxKind.AddExpression:
                    return BinaryOperator.Add;
                case SyntaxKind.SubtractExpression:
                    return BinaryOperator.Subtract;
                case SyntaxKind.MultiplyExpression:
                    return BinaryOperator.Multiply;
                case SyntaxKind.DivideExpression:
                    return BinaryOperator.Divide;
                case SyntaxKind.LessThanExpression:
                    return BinaryOperator.LessThan;
                case SyntaxKind.GreaterThanExpression:
                    return BinaryOperator.GreaterThan;
                case SyntaxKind.LessThanOrEqualExpression:
                    return BinaryOperator.LessThanOrEqual;
                case SyntaxKind.GreaterThanOrEqualExpression:
                    return BinaryOperator.GreaterThanOrEqual;
                case SyntaxKind.BitwiseAndExpression:
                    return BinaryOperator.BitwiseAnd;
                case SyntaxKind.BitwiseOrExpression:
                    return BinaryOperator.BitwiseOr;
                case SyntaxKind.LogicalOrExpression:
                    return BinaryOperator.BooleanOr;
                case SyntaxKind.LogicalAndExpression:
                    return BinaryOperator.BooleanAnd;
                default:
                    throw new NotSupportedException("Unsupported operator: " + kind);
            }
        }

        private IExpression ConvertIdentifierName(IdentifierNameSyntax identifierName)
        {
            var info = model.GetSymbolInfo(identifierName);
            var symbol = info.Symbol;
            if (symbol is ITypeSymbol)
            {
                var tre = Builder.TypeRefExpr();
                tre.Type = (ITypeReference)ConvertTypeReference((ITypeSymbol)symbol);
                return tre;
            }
            if (symbol is ILocalSymbol)
            {
                var varDecl = GetVarDecl((ILocalSymbol)symbol);
                var vre = Builder.VarRefExpr(varDecl);
                return vre;
            }
            if (symbol is IParameterSymbol)
            {
                var paramDecl = GetParamDecl((IParameterSymbol)symbol);
                var vre = Builder.ParamRef(paramDecl);
                return vre;
            }
            if (symbol is IMethodSymbol)
            {
                var methodDecl = GetMethodRef((IMethodSymbol)symbol);
                // TODO get the correct target
                return Builder.MethodRefExpr(methodDecl, Builder.ThisRefExpr());
            }
            if (symbol is IPropertySymbol)
            {
                var propertyDecl = GetPropertyRef((IPropertySymbol)symbol);
                var pre = Builder.PropRefExpr();
                pre.Property = propertyDecl;
                // TODO get the correct target
                pre.Target = Builder.ThisRefExpr();
                return pre;
            }
            throw new NotSupportedException("Unsupported identifier type: " + identifierName);
        }

        private IMethodDeclaration GetMethodRef(IMethodSymbol methodSymbol)
        {
            ITypeDeclaration typeDecl;
            if (typeDecls.TryGetValue(methodSymbol.ContainingType.ToDisplayString(), out typeDecl))
            {
                // TODO hadle overloads
                return typeDecl.Methods.Single(md => md.Name == methodSymbol.Name);
            }

            // TODO construct method ref
            throw new NotSupportedException();
        }

        private IPropertyDeclaration GetPropertyRef(IPropertySymbol propertySymbol)
        {
            // If the prop is on a type that we have a type decl for then return the prop decl
            // else construct and return a prop ref

            ITypeDeclaration typeDecl;
            if (typeDecls.TryGetValue(propertySymbol.ContainingType.ToDisplayString(), out typeDecl))
            {
                return typeDecl.Properties.Single(pd => pd.Name == propertySymbol.Name);
            }

            // TODO construct prop ref
            throw new NotSupportedException();
        }

        Dictionary<string, IVariableDeclaration> vdCache = new Dictionary<string, IVariableDeclaration>();
        private IVariableDeclaration GetVarDecl(ILocalSymbol local)
        {
            // TODO: do this properly
            if (!vdCache.ContainsKey(local.Name))
            {
                throw new Exception("Unknown variable: " + local.Name);
            }
            return vdCache[local.Name];
        }

        Dictionary<string, IParameterDeclaration> pdCache = new Dictionary<string, IParameterDeclaration>();
        private IParameterDeclaration GetParamDecl(IParameterSymbol paramSymbol)
        {
            // TODO: do this properly
            if (!pdCache.ContainsKey(paramSymbol.Name))
            {
                throw new Exception("Unknown parameter: " + paramSymbol.Name);
            }
            return pdCache[paramSymbol.Name];
        }

        private IExpression ConvertMemberAccess(MemberAccessExpressionSyntax memberAccess)
        {
            var symbol = model.GetSymbolInfo(memberAccess.Name).Symbol;
            if (symbol == null)
            {
                throw new NotSupportedException("Couldn't resolve?");
            }

            if (symbol is IMethodSymbol)
            {
                return ConvertMethodReference((IMethodSymbol)symbol, memberAccess);
            }
            if (symbol is IPropertySymbol)
            {
                return ConvertPropertyReference((IPropertySymbol)symbol, memberAccess);
            }
            if (symbol is IFieldSymbol)
            {
                return ConvertFieldReference((IFieldSymbol)symbol, memberAccess);
            }
            if (symbol is INamedTypeSymbol)
            {
                return ConvertNamedType((INamedTypeSymbol)symbol, memberAccess);
            }

            throw new NotSupportedException("Unknown symbol");
        }

        private IExpression ConvertNamedType(INamedTypeSymbol symbol, MemberAccessExpressionSyntax memberAccess)
        {
            var typeRef = (ITypeReference)ConvertTypeReference(symbol);
            return new XTypeReferenceExpression
            {
                Type = typeRef
            };
        }

        private IExpression ConvertFieldReference(IFieldSymbol fieldSymbol, MemberAccessExpressionSyntax memberAccess)
        {
            var fieldRef = new XFieldReference();
            fieldRef.Name = fieldSymbol.Name;
            fieldRef.DeclaringType = ConvertTypeReference(fieldSymbol.ContainingType);
            fieldRef.FieldType = ConvertTypeReference(fieldSymbol.Type);
            // TODO: do this properly
            //fieldRef.MethodInfo = Builder.ToMethodThrows(fieldRef);

            return new XFieldReferenceExpression
            {
                Field = fieldRef,
                Target = ConvertExpression(memberAccess.Expression)
            };
        }

        private IExpression ConvertPropertyReference(IPropertySymbol propSymbol, MemberAccessExpressionSyntax memberAccess)
        {
            var propRef = new XPropertyReference();
            propRef.Name = propSymbol.Name;
            propRef.DeclaringType = ConvertTypeReference(propSymbol.ContainingType);
            propRef.PropertyType = ConvertTypeReference(propSymbol.Type);
            // TODO: do this properly

            return new XPropertyReferenceExpression
            {
                Property = propRef,
                Target = ConvertExpression(memberAccess.Expression)
            };
        }

        private IExpression ConvertMethodReference(IMethodSymbol symbol, MemberAccessExpressionSyntax memberAccess)
        {
            var methodRef = Builder.MethodRef();
            methodRef.Name = symbol.Name;
            methodRef.DeclaringType = ConvertTypeReference(symbol.ContainingType);
            var parTypes = new List<Type>();
            foreach (var par in symbol.Parameters)
            {
                var parameterDecl = ConvertParameter(par);
                methodRef.Parameters.Add(parameterDecl);
                parTypes.Add(Builder.ToType(parameterDecl.ParameterType));
            }
            methodRef.ReturnType = Builder.MethodReturnType(ConvertTypeReference(symbol.ReturnType));
            // TODO genericMethod
            if (symbol.IsGenericMethod)
            {
                var typeArgs = symbol.TypeArguments.Select(ts => ConvertTypeReference(ts)).ToList();
                methodRef.GenericArguments.AddRange(typeArgs);
            }
            // TODO: do this properly
            //methodRef.MethodInfo = Builder.ToType(methodRef.DeclaringType).GetMethod(methodRef.Name, parTypes.ToArray());
            methodRef.MethodInfo = Builder.ToMethodThrows(methodRef);

            return new XMethodReferenceExpression
            {
                Method = methodRef,
                Target = ConvertExpression(memberAccess.Expression)
            };
        }

        private IParameterDeclaration ConvertParameter(IParameterSymbol par)
        {
            var pd = Builder.ParamDecl();
            pd.Name = par.Name;
            pd.ParameterType = ConvertTypeReference(par.Type);
            // TODO: attributes
            return pd;
        }

        private IExpression ConvertLiteral(LiteralExpressionSyntax literalExpression)
        {
            return ConvertConstantExpr(literalExpression);
        }

        private IExpression ConvertConstantExpr(ExpressionSyntax expression)
        {
            var value = model.GetConstantValue(expression);
            var ti = model.GetTypeInfo(expression);
            var val = value.Value;
            if (expression.IsKind(SyntaxKind.NumericLiteralExpression))
            {
                var newType = ConvertTypeReference(ti.ConvertedType).DotNetType;
                val = Convert.ChangeType(val, newType);
            }
            return Builder.LiteralExpr(val);
        }

        private IExpression ConvertInvocation(InvocationExpressionSyntax invocationExpression)
        {
            var c = model.GetConstantValue(invocationExpression);
            if (c.HasValue)
            {
                // compile time constant
                // including nameof(thing)
                return ConvertConstantExpr(invocationExpression);
            }

            var methodExpression = invocationExpression.Expression;
            var argumentList = invocationExpression.ArgumentList.Arguments;
            var convertedArgs = argumentList.Select(arg => ConvertExpression(arg.Expression)).ToList();

            if (argumentList.Any(arg => arg.NameColon != null))
            {
                throw new NotSupportedException("No support yet for named parameters");
            }

            if (model.GetSymbolInfo(invocationExpression).Symbol is IMethodSymbol methodSymbol &&
                methodSymbol.Parameters.Length > 0)
            {
                var lastParam = methodSymbol.Parameters.Last();
                if (lastParam.IsParams && lastParam.Type is IArrayTypeSymbol paramsType)
                {
                    IExpression[] paramsParameters = null;
                    if (argumentList.Count() == methodSymbol.Parameters.Count())
                    {
                        // one element or array supplied?
                        var ti = model.GetTypeInfo(argumentList.Last().Expression);
                        if (!(ti.Type is IArrayTypeSymbol))
                        {
                            paramsParameters = new[] { convertedArgs.Last() };
                        }
                    }
                    else
                    {
                        paramsParameters = convertedArgs.Skip(methodSymbol.Parameters.Length - 1).ToArray();
                    }
                    if (paramsParameters != null)
                    {
                        var ixe = new XArrayCreateExpression();
                        ixe.Type = ConvertTypeReference(paramsType.ElementType);
                        ixe.Dimensions.Add(new XLiteralExpression { Value = paramsParameters.Length });
                        ixe.Initializer = new XBlockExpression();
                        ixe.Initializer.Expressions.AddRange(paramsParameters);

                        convertedArgs.RemoveRange(convertedArgs.Count - paramsParameters.Length, paramsParameters.Length);
                        convertedArgs.Add(ixe);
                    }
                }
            }

            var imie = Builder.MethodInvkExpr();
            imie.Arguments.AddRange(convertedArgs);
            imie.Method = (IMethodReferenceExpression)ConvertExpression(methodExpression);
            return imie;
        }
    }
}
