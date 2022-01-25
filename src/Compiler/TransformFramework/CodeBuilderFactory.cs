// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Gateway to the code model

using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

// Gateway to the code model.

namespace Microsoft.ML.Probabilistic.Compiler
{
    public partial class CodeBuilder
    {
        /// <summary>
        /// Singleton instance
        /// </summary>
        public static CodeBuilder Instance = new CodeBuilder();

        /// <summary>
        /// Default constructor for field declaration
        /// </summary>
        /// <returns>A new field declaration</returns>
        public virtual IFieldDeclaration FieldDecl()
        {
            return new XFieldDeclaration();
        }

        /// <summary>
        /// Default constructor for field reference
        /// </summary>
        /// <returns>A new field reference</returns>
        public virtual IFieldReference FieldRef()
        {
            return new XFieldReference();
        }

        /// <summary>
        /// Default constructor for field reference expression
        /// </summary>
        /// <returns>A new field reference expression</returns>
        public virtual IFieldReferenceExpression FieldRefExpr()
        {
            return new XFieldReferenceExpression();
        }

        /// <summary>
        /// Default constructor for cast expression
        /// </summary>
        /// <returns>A new cast expression</returns>
        public virtual ICastExpression CastExpr()
        {
            return new XCastExpression();
        }

        /// <summary>
        /// Default constructor for checked expression
        /// </summary>
        /// <returns>A new cast expression</returns>
        public virtual ICheckedExpression CheckedExpr()
        {
            return new XCheckedExpression();
        }

        /// <summary>
        /// Default constructor for expression statement
        /// </summary>
        /// <returns>A new expression statement</returns>
        public virtual IExpressionStatement ExprStatement()
        {
            return new XExpressionStatement();
        }

        /// <summary>
        /// Default constructor for a reference type
        /// </summary>
        /// <returns>A new reference type</returns>
        public virtual IReferenceType RefType()
        {
            return new XReferenceType();
        }

        /// <summary>
        /// Default constructor for a type reference
        /// </summary>
        /// <returns>A new type reference</returns>
        public virtual ITypeReference TypeRef()
        {
            return new XTypeReference();
        }

        /// <summary>
        /// Default constructor for type instance reference
        /// </summary>
        /// <returns>A new type instance reference</returns>
        public virtual ITypeReference TypeInstRef()
        {
            return new XTypeInstanceReference();
        }

        /// <summary>
        /// Default constructor for an assembly reference
        /// </summary>
        /// <returns>A new type reference expression</returns>
        public virtual ITypeReferenceExpression TypeRefExpr()
        {
            return new XTypeReferenceExpression();
        }

        /// <summary>
        /// Default constructor for type declaration
        /// </summary>
        /// <returns>A new type declaration</returns>
        public virtual ITypeDeclaration TypeDecl()
        {
            return new XTypeDeclaration();
        }

        /// <summary>
        /// Check if ITypeReference is TypeInstanceReference
        /// </summary>
        /// <returns>true if it is a TypeInstanceReference</returns>
        public virtual bool IsTypeInstRef(ITypeReference itr)
        {
            return (itr is XTypeInstanceReference);
        }

        /// <summary>
        /// Default constructor for an assembly reference
        /// </summary>
        /// <returns>A new assembly reference</returns>
        public virtual IAssemblyReference AssemblyRef()
        {
            return new XAssemblyReference();
        }

        /// <summary>
        /// Default constructor for address out expression
        /// </summary>
        /// <returns>A new address out expression</returns>
        public virtual IAddressOutExpression AddrOutExpr()
        {
            return new XAddressOutExpression();
        }

        /// <summary>
        /// Default constructor for address ref expression
        /// </summary>
        /// <returns>A new address ref expression</returns>
        public virtual IAddressReferenceExpression AddrRefExpr()
        {
            return new XAddressReferenceExpression();
        }

        /// <summary>
        /// Default constructor for an address dereference expression
        /// </summary>
        /// <returns></returns>
        public virtual IAddressDereferenceExpression AddrDerefExpr()
        {
            return new XAddressDereferenceExpression();
        }

        /// <summary>
        /// Default constructor for method reference
        /// </summary>
        /// <returns>A new method reference</returns>
        public virtual IMethodReference MethodRef()
        {
            return new XMethodReference();
        }

        /// <summary>
        /// Default constructor for method instance reference
        /// </summary>
        /// <returns>A new method reference</returns>
        public virtual IMethodReference MethodInstRef()
        {
            return new XMethodInstanceReference();
        }

        /// <summary>
        /// Constructor for method instance reference
        /// </summary>
        /// <returns>A new method reference</returns>
        public virtual IMethodReference MethodInstRef(IType[] paramTypes)
        {
            XMethodInstanceReference xmir = new XMethodInstanceReference();
            if (paramTypes != null)
            {
                for (int i = 0; i < paramTypes.Length; i++)
                {
                    XParameterDeclaration xpd = new XParameterDeclaration();
                    xpd.ParameterType = paramTypes[i];
                    xmir.Parameters.Add(xpd);
                }
            }
            return xmir;
        }

        /// <summary>
        /// Check if IMethodReference is MethodInstanceReference
        /// </summary>
        /// <returns>true if MathodInstanceReference</returns>
        public virtual bool IsMethodInstRef(IMethodReference imr)
        {
            return (imr is XMethodInstanceReference);
        }

        /// <summary>
        /// Default constructor for method return type
        /// </summary>
        /// <returns>A new method return type</returns>
        public virtual IMethodReturnType MethodReturnType(IType type)
        {
            IMethodReturnType imrt = new XMethodReturnType();
            imrt.Type = type;
            return imrt;
        }

        /// <summary>
        /// Default constructor for method declaration
        /// </summary>
        /// <returns>A new method declaration</returns>
        public virtual IMethodDeclaration MethodDecl()
        {
            return new XMethodDeclaration();
        }

        /// <summary>
        /// Default constructor for method return statement
        /// </summary>
        /// <returns>A new method return statement</returns>
        public virtual IMethodReturnStatement MethodRtrnStmt()
        {
            return new XMethodReturnStatement();
        }

        /// <summary>
        /// Default constructor for method invoke expression
        /// </summary>
        /// <returns>A new method invoke expression</returns>
        public virtual IMethodInvokeExpression MethodInvkExpr()
        {
            return new XMethodInvokeExpression();
        }

        /// <summary>
        /// Default constructor for method reference expression
        /// </summary>
        /// <returns>A new method reference expression</returns>
        public virtual IMethodReferenceExpression MethodRefExpr(IMethodReference method, IExpression target)
        {
            return new XMethodReferenceExpression()
            {
                Method = method,
                Target = target
            };
        }

        /// <summary>
        /// Default constructor for block statement
        /// </summary>
        /// <returns>A new block statement</returns>
        public virtual IBlockStatement BlockStmt()
        {
            return new XBlockStatement();
        }

        /// <summary>
        /// Default constructor for block expression
        /// </summary>
        /// <returns>A new block expression</returns>
        public virtual IBlockExpression BlockExpr()
        {
            return new XBlockExpression();
        }

        /// <summary>
        /// Default constructor for literal expression
        /// </summary>
        /// <returns>A new literal expression</returns>
        public virtual ILiteralExpression LiteralExpr()
        {
            return new XLiteralExpression();
        }

        /// <summary>
        /// Default constructor for binary expression
        /// </summary>
        /// <returns>A new binary expression</returns>
        public virtual IBinaryExpression BinaryExpr()
        {
            return new XBinaryExpression();
        }

        /// <summary>
        /// Default constructor for typeof expression
        /// </summary>
        /// <returns>A new typeof expression</returns>
        public virtual ITypeOfExpression TypeOfExpr()
        {
            return new XTypeOfExpression();
        }

        /// <summary>
        /// Default constructor for object create expression
        /// </summary>
        /// <returns>A new object create expression</returns>
        public virtual IObjectCreateExpression ObjCreateExpr()
        {
            return new XObjectCreateExpression();
        }

        /// <summary>
        /// Default constructor for variable reference expression
        /// </summary>
        /// <returns>A new variable reference expression</returns>
        public virtual IVariableReferenceExpression VarRefExpr()
        {
            return new XVariableReferenceExpression();
        }

        /// <summary>
        /// Constructor for variable reference expression array
        /// </summary>
        /// <param name="count">Number of variable reference expressions in the array</param>
        /// <returns>A new variable reference expression array </returns>
        public virtual IVariableReferenceExpression[] VarRefExprArray(int count)
        {
            return new XVariableReferenceExpression[count];
        }

        /// <summary>
        /// Default constructor for variable declaration
        /// </summary>
        /// <returns>A new variable declaration</returns>
        public virtual IVariableDeclaration VarDecl()
        {
            return new XVariableDeclaration();
        }

        /// <summary>
        /// Default constructor for variable declaration expression
        /// </summary>
        /// <returns>A new variable declaration expression</returns>
        public virtual IVariableDeclarationExpression VarDeclExpr()
        {
            return new XVariableDeclarationExpression();
        }

        /// <summary>
        /// Constructor for variable reference
        /// </summary>
        /// <param name="ivd">Variable declaration interface instance</param>
        /// <returns>new variable reference</returns>
        public virtual IVariableReference VarRef(IVariableDeclaration ivd)
        {
            IVariableReference ivr = new XVariableReference();
            ivr.Variable = ivd;
            return ivr;
        }

        /// <summary>
        /// Default constructor for assign expression
        /// </summary>
        /// <returns>A new assign expression</returns>
        public virtual IAssignExpression AssignExpr()
        {
            return new XAssignExpression();
        }

        /// <summary>
        /// Default constructor for comment statement
        /// </summary>
        /// <returns>A new comment statement</returns>
        public virtual ICommentStatement CommentStmt()
        {
            return new XCommentStatement();
        }

        /// <summary>
        /// Default constructor for comment
        /// </summary>
        /// <returns>A new comment</returns>
        public virtual IComment Comment()
        {
            return new XComment();
        }

        /// <summary>
        /// Default constructor for argument reference expression
        /// </summary>
        /// <returns>A new argument reference expression</returns>
        public virtual IArgumentReferenceExpression ParamRef()
        {
            return new XArgumentReferenceExpression();
        }

        /// <summary>
        /// Default constructor for parameter declaration
        /// </summary>
        /// <returns>A new parameter declaration</returns>
        public virtual IParameterDeclaration ParamDecl()
        {
            return new XParameterDeclaration();
        }

        /// <summary>
        /// Default constructor for property declaration
        /// </summary>
        /// <returns>A new property declaration</returns>
        public virtual IPropertyDeclaration PropDecl()
        {
            return new XPropertyDeclaration();
        }

        /// <summary>
        /// Default constructor for property reference
        /// </summary>
        /// <returns>A new property reference</returns>
        public virtual IPropertyReference PropRef()
        {
            return new XPropertyReference();
        }

        /// <summary>
        /// Default constructor for property reference expression
        /// </summary>
        /// <returns>A new property reference expression</returns>
        public virtual IPropertyReferenceExpression PropRefExpr()
        {
            return new XPropertyReferenceExpression();
        }

        /// <summary>
        /// Default constructor for property indexer expression
        /// </summary>
        /// <returns>A new property indexer expression</returns>
        public virtual IPropertyIndexerExpression PropIndxrExpr()
        {
            return new XPropertyIndexerExpression();
        }

        /// <summary>
        /// Default constructor for event declaration
        /// </summary>
        /// <returns>A new event declaration</returns>
        public virtual IEventDeclaration EventDecl()
        {
            return new XEventDeclaration();
        }

        /// <summary>
        /// Default constructor for event reference expression
        /// </summary>
        /// <returns>A new event reference expression</returns>
        public virtual IEventReferenceExpression EventRefExpr()
        {
            return new XEventReferenceExpression();
        }

        /// <summary>
        /// Default constructor for a delegate invoke expression
        /// </summary>
        /// <returns></returns>
        public virtual IDelegateInvokeExpression DelegateInvokeExpr()
        {
            return new XDelegateInvokeExpression();
        }

        /// <summary>
        /// Default constructor for a for statement
        /// </summary>
        /// <returns>A new for statement</returns>
        public virtual IForStatement ForStmt()
        {
            return new XForStatement();
        }

        /// <summary>
        /// Default constructor for a repeat statement
        /// </summary>
        /// <returns>A new repeat statement</returns>
        public virtual IRepeatStatement RepeatStmt()
        {
            return new XRepeatStatement();
        }

        /// <summary>
        /// Default constructor for foreach statement
        /// </summary>
        /// <returns>A new foreach statement</returns>
        public virtual IForEachStatement ForEachStmt()
        {
            return new XForEachStatement();
        }

        /// <summary>
        /// Default constructor for unary expression
        /// </summary>
        /// <returns>A new unary expression</returns>
        public virtual IUnaryExpression UnaryExpr()
        {
            return new XUnaryExpression();
        }

        /// <summary>
        /// Default constructor for condition statement
        /// </summary>
        /// <returns>A new condition statement</returns>
        public virtual IConditionStatement CondStmt()
        {
            return new XConditionStatement();
        }

        /// <summary>
        /// Default constructor for switch statement
        /// </summary>
        /// <returns>A new switch statement</returns>
        public virtual ISwitchStatement SwitchStmt()
        {
            return new XSwitchStatement();
        }

        /// <summary>
        /// Default constructor for break statement
        /// </summary>
        /// <returns>A new break statement</returns>
        public virtual IBreakStatement BreakStmt()
        {
            return new XBreakStatement();
        }

        /// <summary>
        /// Default constructor for while statement
        /// </summary>
        /// <returns>A new while statement</returns>
        public virtual IWhileStatement WhileStmt()
        {
            return new XWhileStatement();
        }

        /// <summary>
        /// Default constructor for using statement
        /// </summary>
        /// <returns>A new statement</returns>
        public virtual IUsingStatement UsingStmt()
        {
            return new XUsingStatement();
        }

        /// <summary>
        /// Default constructor for 'this' refrence expression
        /// </summary>
        /// <returns>A new this reference expression</returns>
        public virtual IThisReferenceExpression ThisRefExpr()
        {
            return new XThisReferenceExpression();
        }

        /// <summary>
        /// Default constructor for statement collection
        /// </summary>
        /// <returns>A new statement collection</returns>
        public virtual IList<IStatement> StmtCollection()
        {
            return new List<IStatement>();
        }

        /// <summary>
        /// Default constructor for expression collection
        /// </summary>
        /// <returns>A new expression collection</returns>
        public virtual IList<IExpression> ExprCollection()
        {
            return new List<IExpression>();
        }

        /// <summary>
        /// Default constructor for namespace
        /// </summary>
        /// <returns>A new namespace</returns>
        public virtual INamespace NameSpace()
        {
            return new XNamespace();
        }

        /// <summary>
        /// Default constructor for base reference expression
        /// </summary>
        /// <returns>A new base reference expression</returns>
        public virtual IBaseReferenceExpression BaseRefExpr()
        {
            return new XBaseReferenceExpression();
        }

        /// <summary>
        /// Default constructor for condition expression
        /// </summary>
        /// <returns>A new condition expression</returns>
        public virtual IConditionExpression CondExpr()
        {
            return new XConditionExpression();
        }

        /// <summary>
        /// Default constructor for constructor declaration
        /// </summary>
        /// <returns>A new constructor declaration</returns>
        public virtual IConstructorDeclaration ConstructorDecl()
        {
            return new XConstructorDeclaration();
        }

        /// <summary>
        /// Default constructor for condition case
        /// </summary>
        /// <returns>A new condition case</returns>
        public virtual IConditionCase CondCase()
        {
            return new XConditionCase();
        }

        /// <summary>
        /// Default constructor for default case
        /// </summary>
        /// <returns>A new default case</returns>
        public virtual IDefaultCase DefCase()
        {
            return new XDefaultCase();
        }

        /// <summary>
        /// Default constructor for array type
        /// </summary>
        /// <returns>A new array type</returns>
        public virtual IArrayType ArrayType()
        {
            return new XArrayType();
        }

        /// <summary>
        /// Default constructor for array create expression
        /// </summary>
        /// <returns>A new array create expression</returns>
        public virtual IArrayCreateExpression ArrayCreateExpr()
        {
            return new XArrayCreateExpression();
        }

        /// <summary>
        /// Default constructor for array indexer expression
        /// </summary>
        /// <returns>A new array indexer expression</returns>
        public virtual IArrayIndexerExpression ArrayIndxrExpr()
        {
            return new XArrayIndexerExpression();
        }

        /// <summary>
        /// Default constructor for anonymous method expression
        /// </summary>
        /// <returns>A new anonymous method expression</returns>
        public virtual IAnonymousMethodExpression AnonMethodExpr()
        {
            return new XAnonymousMethodExpression();
        }

        /// <summary>
        /// Throw exception statement
        /// </summary>
        /// <returns></returns>
        public virtual IThrowExceptionStatement ThrowStmt()
        {
            return new XThrowExceptionStatement();
        }

        /// <summary>
        /// Default(T) expression
        /// </summary>
        /// <returns></returns>
        public virtual IDefaultExpression DefaultExpr()
        {
            return new XDefaultExpression();
        }

        /// <summary>
        /// Creates a generic type parameter
        /// </summary>
        /// <param name="name">Type parameter name</param>
        /// <returns></returns>
        public IGenericParameter GenericTypeParam(string name)
        {
            var tp = new XGenericParameter();
            tp.Name = name;
            return tp;
        }
    }
}