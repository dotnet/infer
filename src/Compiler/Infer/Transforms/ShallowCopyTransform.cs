// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// A transform which produces a copy where subtrees from the original are reused if they do not change.
    /// Any empty containers are removed.
    /// </summary>
    public class ShallowCopyTransform : ICodeTransform, IExpressionTransform
    {
        public virtual string Name
        {
            get { return "ShallowCopyTransform"; }
        }

        /// <summary>
        /// If true, statements will be shallow copied even if they do not change.  Child statements need not be copied.
        /// </summary>
        public bool ShallowCopy;

        /// <summary>
        /// Holds contextual information about the state of the transform which may be used
        /// to affect the transform.
        /// </summary>
        protected BasicTransformContext context = new BasicTransformContext();

        public virtual ICodeTransformContext Context
        {
            get { return context; }
            set { context = (BasicTransformContext)value; }
        }

        /// <summary>
        /// Helps build class declarations
        /// </summary>
        protected static CodeBuilder Builder = CodeBuilder.Instance;

        /// <summary>
        /// Helps recognize code patterns
        /// </summary>
        protected static CodeRecognizer Recognizer = CodeRecognizer.Instance;

        protected virtual void Initialise()
        {
            context.Results.Transform = this;
        }

        public virtual ITypeDeclaration Transform(ITypeDeclaration itd)
        {
            Initialise();
            ITypeDeclaration td = ConvertType(itd);
            return td;
        }

        /// <summary>
        /// Convert a type declaration
        /// </summary>
        /// 
        /// <param name="itd">The type to convert</param>
        /// <returns>A new type declaration</returns>
        public virtual ITypeDeclaration ConvertType(ITypeDeclaration itd)
        {
            Context.OpenType(itd);
            ITypeDeclaration td = Builder.TypeDecl();
            context.SetPrimaryOutput(td);
            ConvertTypeProperties(td, itd);

            // methods
            ConvertMethods(td, itd);
            // fields
            ConvertFields(td, itd);
            // properties
            ConvertProperties(td, itd);

            //td.Attributes.AddRange(itd.Attributes); // todo: convert
            ConvertEvents(td, itd);
            td.GenericArguments.AddRange(itd.GenericArguments);

            // nested types
            ConvertNestedTypes(td, itd);

            Context.CloseType(itd);
            if (TypeMembersAreReferenceEqual(td, itd)) return itd;
            return td;
        }

        private bool TypeMembersAreReferenceEqual(ITypeDeclaration td, ITypeDeclaration itd)
        {
            if (td.Abstract != itd.Abstract) return false;
            if (!ItemsAreReferenceEqual(td.Attributes, itd.Attributes)) return false;
            if (!ReferenceEquals(td.BaseType, itd.BaseType)) return false;
            if (td.Documentation != itd.Documentation) return false;
            if (!ReferenceEquals(td.DotNetType, itd.DotNetType)) return false;
            if (!ItemsAreReferenceEqual(td.Events, itd.Events)) return false;
            if (!ItemsAreReferenceEqual(td.Fields, itd.Fields)) return false;
            if (!ItemsAreReferenceEqual(td.GenericArguments, itd.GenericArguments)) return false;
            if (!ReferenceEquals(td.GenericType, itd.GenericType)) return false;
            if (td.Interface != itd.Interface) return false;
            if (!ItemsAreReferenceEqual(td.Interfaces, itd.Interfaces)) return false;
            if (!ItemsAreReferenceEqual(td.Methods, itd.Methods)) return false;
            if (td.Name != itd.Name) return false;
            if (td.Namespace != itd.Namespace) return false;
            if (!ItemsAreReferenceEqual(td.NestedTypes, itd.NestedTypes)) return false;
            if (!ReferenceEquals(td.Owner, itd.Owner)) return false;
            if (!ItemsAreReferenceEqual(td.Properties, itd.Properties)) return false;
            if (td.Sealed != itd.Sealed) return false;
            if (td.Partial != itd.Partial) return false;
            //if (td.ValueType != itd.ValueType) return false;
            if (td.Visibility != itd.Visibility) return false;
            return true;
        }

        public virtual void ConvertTypeProperties(ITypeDeclaration td, ITypeDeclaration itd)
        {
            td.Abstract = itd.Abstract;
            // attributes
            td.BaseType = itd.BaseType; // ConvertBaseType(itd.BaseType);
            td.Documentation = itd.Documentation;
            // events
            // generic arguments
            // td.GenericType = itd.GenericType; - not supported
            foreach (ITypeReference tr in itd.Interfaces)
            {
                ITypeReference itr = ConvertInterface(tr);
                if (itr == null) continue;
                td.Interfaces.Add(itr);
            }
            // interfaces
            td.Name = CheckIdentifier(itd.Name);
            td.Namespace = CheckIdentifier(itd.Namespace);
            // nested types
            td.Owner = itd.Owner;
            // properties
            td.Sealed = itd.Sealed;
            td.Partial = itd.Partial;
            td.Visibility = itd.Visibility;
            td.DotNetType = itd.DotNetType;
            context.InputAttributes.CopyObjectAttributesTo(itd, context.OutputAttributes, td);
        }

        protected virtual ITypeReference ConvertBaseType(ITypeReference itr)
        {
            if (itr == null) return null;
            return ConvertTypeReference(itr);
        }

        protected virtual ITypeReference ConvertInterface(ITypeReference itr)
        {
            return itr; // ConvertTypeReference(itr);
        }

        protected virtual void ConvertNestedTypes(ITypeDeclaration td, ITypeDeclaration itd)
        {
            foreach (ITypeDeclaration itd2 in itd.NestedTypes)
            {
                ITypeDeclaration td2 = ConvertType(itd2);
                if (td2 != null)
                {
                    td2.Owner = td;
                    td.NestedTypes.Add(td2);
                }
            }
        }

        protected virtual void ConvertMethods(ITypeDeclaration td, ITypeDeclaration itd)
        {
            foreach (IMethodDeclaration imd in itd.Methods)
            {
                IMethodDeclaration md = ConvertMethod(imd);
                if (md != null) td.Methods.Add(md);
            }
        }

        protected virtual IMethodDeclaration ConvertMethod(IMethodDeclaration imd)
        {
            context.OpenMember(imd);
            if (imd is IConstructorDeclaration icd)
            {
                IConstructorDeclaration cd = Builder.ConstructorDecl();
                context.SetPrimaryOutput(cd);
                IMethodDeclaration cd2 = DoConvertConstructor(cd, icd);
                context.CloseMember(imd);
                if (cd2 != null) context.InputAttributes.CopyObjectAttributesTo(imd, context.OutputAttributes, cd2);
                return cd2;
            }
            IMethodDeclaration md = Builder.MethodDecl();
            context.SetPrimaryOutput(md);
            IMethodDeclaration md2 = DoConvertMethod(md, imd);
            context.CloseMember(imd);
            if (md2 != null) context.InputAttributes.CopyObjectAttributesTo(imd, context.OutputAttributes, md2);
            return md2;
        }

        protected virtual void ConvertEvents(ITypeDeclaration td, ITypeDeclaration itd)
        {
            foreach (IEventDeclaration ifd in itd.Events)
            {
                context.OpenMember(ifd);
                IEventDeclaration fd = ConvertEvent(td, ifd);
                if (fd != null)
                {
                    context.AddMember(fd);
                    td.Events.Add(fd);
                }
                context.CloseMember(ifd);
            }
        }

        protected virtual IEventDeclaration ConvertEvent(ITypeDeclaration td, IEventDeclaration ifd)
        {
            bool shallow = true;
            if (shallow) return ifd;
            else
            {
                IEventDeclaration fd = Builder.EventDecl();
                fd.Attributes.AddRange(ifd.Attributes);
                fd.DeclaringType = td;
                fd.Documentation = ifd.Documentation;
                fd.EventType = ifd.EventType;
                fd.GenericEvent = ifd.GenericEvent;
                fd.InvokeMethod = ConvertMethodReference(ifd.InvokeMethod);
                fd.Name = CheckIdentifier(ifd.Name);
                return fd;
            }
        }

        protected virtual void ConvertFields(ITypeDeclaration td, ITypeDeclaration itd)
        {
            foreach (IFieldDeclaration ifd in itd.Fields)
            {
                context.OpenMember(ifd);
                IFieldDeclaration fd = ConvertField(td, ifd);
                if (fd != null)
                {
                    context.AddMember(fd);
                    td.Fields.Add(fd);
                }
                context.CloseMember(ifd);
            }
        }

        protected virtual IFieldDeclaration ConvertField(ITypeDeclaration td, IFieldDeclaration ifd)
        {
            bool shallow = true;
            if (shallow) return ifd;
            else
            {
                IFieldDeclaration fd = Builder.FieldDecl();
                fd.Attributes.AddRange(ifd.Attributes);
                fd.DeclaringType = td;
                fd.Documentation = ifd.Documentation;
                fd.FieldType = ifd.FieldType;
                fd.Initializer = ifd.Initializer;
                fd.Literal = ifd.Literal;
                fd.Name = CheckIdentifier(ifd.Name);
                fd.ReadOnly = ifd.ReadOnly;
                fd.Visibility = ifd.Visibility;
                fd.Static = ifd.Static;
                return fd;
            }
        }

        protected virtual void ConvertProperties(ITypeDeclaration td, ITypeDeclaration itd)
        {
            foreach (IPropertyDeclaration ipd in itd.Properties)
            {
                context.OpenMember(ipd);
                IPropertyDeclaration pd = ConvertProperty(td, ipd);
                if (pd != null)
                {
                    context.AddMember(pd);
                    td.Properties.Add(pd);
                }
                context.CloseMember(ipd);
            }
        }

        protected virtual IPropertyDeclaration ConvertProperty(ITypeDeclaration td, IPropertyDeclaration ipd, bool convertGetterAndSetter = true)
        {
            IPropertyDeclaration pd = Builder.PropDecl();
            context.SetPrimaryOutput(pd);
            pd.Attributes.AddRange(ipd.Attributes);
            pd.DeclaringType = td;
            pd.Documentation = ipd.Documentation;
            pd.Initializer = ipd.Initializer;
            pd.Name = CheckIdentifier(ipd.Name);
            pd.PropertyType = ipd.PropertyType;
            if (convertGetterAndSetter)
            {
                if (ipd.GetMethod != null) pd.GetMethod = ConvertMethod((IMethodDeclaration)ipd.GetMethod);
                if (ipd.SetMethod != null) pd.SetMethod = ConvertMethod((IMethodDeclaration)ipd.SetMethod);
            }
            if (ReferenceEquals(pd.GetMethod, ipd.GetMethod) && ReferenceEquals(pd.SetMethod, ipd.SetMethod)) return ipd;
            return pd;
        }

        protected virtual IMethodDeclaration DoConvertConstructor(IConstructorDeclaration cd, IConstructorDeclaration icd)
        {
            cd.Initializer = (IMethodInvokeExpression)ConvertExpression(icd.Initializer);
            return DoConvertMethod(cd, icd);
        }

        protected virtual IMethodDeclaration DoConvertMethod(IMethodDeclaration md, IMethodDeclaration imd)
        {
            md.Abstract = imd.Abstract;
            md.Attributes.AddRange(imd.Attributes);
            md.DeclaringType = context.FindOutputForAncestor<ITypeDeclaration, ITypeDeclaration>();
            md.Documentation = imd.Documentation;
            md.Final = imd.Final;
            md.GenericArguments.AddRange(imd.GenericArguments);
            md.Name = CheckIdentifier(imd.Name);
            md.MethodInfo = imd.MethodInfo;
            md.Overrides = imd.Overrides;
            for (int i = 0; i < imd.Parameters.Count; i++)
            {
                IParameterDeclaration ipd = ConvertMethodParameter(imd.Parameters[i], i);
                if (ipd != null) md.Parameters.Add(ipd);
            }
            IMethodReturnType mrt = Builder.MethodReturnType(imd.ReturnType.Type);
            mrt.Attributes.AddRange(imd.ReturnType.Attributes);
            md.ReturnType = mrt;
            md.Visibility = imd.Visibility;
            md.Static = imd.Static;
            md.Virtual = imd.Virtual;
            context.InputAttributes.CopyObjectAttributesTo(imd, context.OutputAttributes, md);
            IBlockStatement inputBody = imd.Body;
            Context.OpenStatement(inputBody);
            IBlockStatement outputBody = Builder.BlockStmt();
            context.SetPrimaryOutput(outputBody);
            DoConvertMethodBody(outputBody.Statements, inputBody.Statements);
            Context.CloseStatement(inputBody);
            md.Body = outputBody;
            if (ItemsAreReferenceEqual(md.Body.Statements, imd.Body.Statements)) return imd;
            return md;
        }

        public static string CheckIdentifier(string mname)
        {
            if (mname.Contains("@")) mname = mname.Replace('@', '_');
            if (mname.Contains("<")) mname = mname.Replace('<', '_');
            if (mname.Contains(">")) mname = mname.Replace('>', '_');
            return mname;
        }

        protected virtual void DoConvertMethodBody(IList<IStatement> outputs, IList<IStatement> inputs)
        {
            ConvertStatements(outputs, inputs);
        }

        protected virtual IParameterDeclaration ConvertMethodParameter(IParameterDeclaration ipd, int index)
        {
            // must not convert declarations by default, else references will get messed up
            return ipd;
        }

        protected virtual IBlockStatement ConvertBlock(IBlockStatement inputBlock)
        {
            Context.OpenStatement(inputBlock);
            IBlockStatement outputBlock = ConvertBlockAlreadyOpen(inputBlock);
            Context.CloseStatement(inputBlock);
            return outputBlock;
        }

        // This is a temporary fix to make the transform browser work properly for shallow copies
        protected void RegisterUnchangedStatements(IEnumerable<IStatement> stmts)
        {
            foreach (IStatement st in stmts)
            {
                RegisterUnchangedStatement(st);
            }
        }

        protected void RegisterUnchangedStatement(IStatement st)
        {
            context.OpenStatement(st);
            context.SetPrimaryOutput(st);
            context.CloseStatement(st);
        }

        protected Stack<ICollection<IStatement>> beforeStack = new Stack<ICollection<IStatement>>();
        protected Stack<List<IStatement>> afterStack = new Stack<List<IStatement>>();

        protected void OpenOutputBlock(ICollection<IStatement> outputs)
        {
            beforeStack.Push(outputs);
            List<IStatement> afterStmts = new List<IStatement>();
            afterStack.Push(afterStmts);
        }

        /// <summary>
        /// When using OpenOutputBlock, this must be called after converting each statement.
        /// </summary>
        protected void FinishConvertStatement()
        {
            ICollection<IStatement> outputs = beforeStack.Peek();
            List<IStatement> afterStmts = afterStack.Peek();
            outputs.AddRange(afterStmts);
            afterStmts.Clear();
        }

        protected void CloseOutputBlock()
        {
            afterStack.Pop();
            beforeStack.Pop();
        }

        protected virtual void ConvertStatements(IList<IStatement> outputs, IEnumerable<IStatement> inputs)
        {
            OpenOutputBlock(outputs);
            foreach (IStatement ist in inputs)
            {
                IStatement st = ConvertStatement(ist);
                if (st != null) outputs.Add(st);
                FinishConvertStatement();
            }
            CloseOutputBlock();
        }

        internal void ProcessPending(List<AddAction> toAddBefore, List<AddAction> toAddAfterwards)
        {
            if (afterStack.Count == 0)
            {
                if (toAddBefore != null || toAddAfterwards != null)
                    throw new Exception("Cannot add statements to an empty block stack");
                return;
            }
            List<IStatement> afterStmts = afterStack.Peek();
            ICollection<IStatement> beforeStmts = beforeStack.Peek();
            // while we have statements which we need to convert and then add
            if (toAddBefore != null)
            {
                // convert each statement in the collection and add it before this statement
                foreach (AddAction a in toAddBefore)
                {
                    if (a.DoConversion)
                    {
                        // TODO: need to fix up TransformOutput. changing this statement breaks the association in TransformOutput.
                        IStatement st = ConvertStatement((IStatement)a.Object);
                        if (st != null) beforeStmts.Add(st);
                    }
                    else beforeStmts.Add((IStatement)a.Object);
                }
            }
            if (toAddAfterwards != null)
            {
                // convert each statement in the collection and add it after this statement
                foreach (AddAction a in toAddAfterwards)
                {
                    if (a.DoConversion)
                    {
                        IStatement st = ConvertStatement((IStatement)a.Object);
                        if (st != null) afterStmts.Add(st);
                    }
                    else afterStmts.Add((IStatement)a.Object);
                }
            }
        }

        protected virtual IStatement ConvertStatement(IStatement ist)
        {
            Context.OpenStatement(ist);
            IStatement newStmt = DoConvertStatement(ist);
            TransformInfo ti = context.InputStack[context.InputStack.Count - 1];
            if (newStmt != null && ti.PrimaryOutput == null) context.SetPrimaryOutput(newStmt);
            List<AddAction> toAddBefore = ti.toAddBefore;
            List<AddAction> toAddAfterwards = ti.toAddAfterwards;
            context.CloseStatement(ist);
            ProcessPending(toAddBefore, toAddAfterwards);
            return newStmt;
        }

        protected virtual IStatement DoConvertStatement(IStatement ist)
        {
            if (ist is IBlockStatement ibs)
            {
                return ConvertBlockAlreadyOpen(ibs);
            }
            else if (ist is IExpressionStatement ies)
            {
                return ConvertExpressionStatement(ies);
            }
            else if (ist is IMethodReturnStatement imrs)
            {
                return ConvertReturnStatement(imrs);
            }
            else if (ist is IConditionStatement ics)
            {
                return ConvertCondition(ics);
            }
            else if (ist is IForStatement ifs)
            {
                return ConvertFor(ifs);
            }
            else if (ist is IRepeatStatement irs)
            {
                return ConvertRepeat(irs);
            }
            else if (ist is IWhileStatement iws)
            {
                return ConvertWhile(iws);
            }
            else if (ist is IForEachStatement ifes)
            {
                return ConvertForEach(ifes);
            }
            else if (ist is IUsingStatement ius)
            {
                return ConvertUsing(ius);
            }
            else if (ist is ICommentStatement icms)
            {
                return ConvertComment(icms);
            }
            else if (ist is ISwitchStatement iss)
            {
                return ConvertSwitch(iss);
            }
            else if (ist is IBreakStatement ibrs)
            {
                return ConvertBreak(ibrs);
            }
            else if (ist is IThrowExceptionStatement ites)
            {
                return ConvertThrow(ites);
            }
            else throw new Exception("Unhandled statement: " + ist.GetType());
        }

        protected virtual IStatement ConvertThrow(IThrowExceptionStatement its)
        {
            return its;
        }

        protected virtual IStatement ConvertBreak(IBreakStatement ibs)
        {
            return ibs;
        }

        protected virtual IStatement ConvertSwitch(ISwitchStatement iss)
        {
            ISwitchStatement ss = Builder.SwitchStmt();
            context.SetPrimaryOutput(ss);
            ss.Expression = ConvertExpression(iss.Expression);
            foreach (ISwitchCase isc in iss.Cases)
            {
                ConvertSwitchCase(ss.Cases, isc);
            }
            return ss;
        }

        protected virtual void ConvertSwitchCase(IList<ISwitchCase> cases, ISwitchCase isc)
        {
            ISwitchCase isc2;
            if (isc is IConditionCase icc)
            {
                IConditionCase cc = Builder.CondCase();
                cc.Condition = ConvertExpression(icc.Condition);
                isc2 = cc;
            }
            else
            {
                if (!(isc is IDefaultCase)) throw new InferCompilerException("Unexpected switch case type: " + isc.GetType());
                isc2 = Builder.DefCase();
            }
            isc2.Body = ConvertBlock(isc.Body);
            cases.Add(isc2);
        }

        protected virtual IStatement ConvertUsing(IUsingStatement ius)
        {
            IUsingStatement us = Builder.UsingStmt();
            context.SetPrimaryOutput(us);
            us.Expression = ConvertExpression(ius.Expression);
            us.Body = ConvertBlock(ius.Body);
            return us;
        }

        protected virtual IStatement ConvertWhile(IWhileStatement iws)
        {
            IWhileStatement ws = Builder.WhileStmt(iws);
            context.SetPrimaryOutput(ws);
            bool wasCopy = ShallowCopy;
            ShallowCopy = false;
            ws.Condition = ConvertExpression(iws.Condition);
            ShallowCopy = wasCopy;
            ws.Body = ConvertBlock(iws.Body);
            context.InputAttributes.CopyObjectAttributesTo(iws, context.OutputAttributes, ws);
            return ws;
        }

        protected virtual IForEachStatement ConvertForEach(IForEachStatement ifs)
        {
            IForEachStatement fs = Builder.ForEachStmt();
            context.SetPrimaryOutput(fs);
            fs.Expression = ConvertExpression(ifs.Expression);
            IVariableDeclarationExpression vde = Builder.VarDeclExpr();
            vde.Variable = ifs.Variable;
            IVariableDeclarationExpression vde2 = (IVariableDeclarationExpression)ConvertExpression(vde);
            fs.Variable = vde2.Variable;
            fs.Body = ConvertBlock(ifs.Body);
            context.InputAttributes.CopyObjectAttributesTo(ifs, context.OutputAttributes, fs);
            return fs;
        }

        public virtual IExpression ConvertExpression(IExpression expr)
        {
            context.OpenExpression(expr);
            IExpression expr2 = DoConvertExpression(expr);
            context.CloseExpression(expr);
            return expr2;
        }

        protected virtual IExpression DoConvertExpression(IExpression expr)
        {
            if (expr is IAssignExpression iae) return ConvertAssign(iae);
            if (expr is IMethodInvokeExpression imie) return ConvertMethodInvoke(imie);
            if (expr is IMethodReferenceExpression imre) return ConvertMethodRefExpr(imre);
            if (expr is IUnaryExpression iue) return ConvertUnary(iue);
            if (expr is ILiteralExpression ile) return ConvertLiteral(ile);
            if (expr is IVariableDeclarationExpression ivde) return ConvertVariableDeclExpr(ivde);
            if (expr is IVariableReferenceExpression ivre) return ConvertVariableRefExpr(ivre);
            if (expr is IArrayIndexerExpression iaie) return ConvertArrayIndexer(iaie);
            if (expr is IArrayCreateExpression iace) return ConvertArrayCreate(iace);
            if (expr is IArgumentReferenceExpression iare) return ConvertArgumentRef(iare);
            if (expr is IFieldReferenceExpression ifre) return ConvertFieldRefExpr(ifre);
            if (expr is IBinaryExpression ibe) return ConvertBinary(ibe);
            if (expr is IPropertyReferenceExpression ipre) return ConvertPropertyRefExpr(ipre);
            if (expr is IThisReferenceExpression ithis) return ConvertThis(ithis);
            if (expr is IBlockExpression ible) return ConvertBlockExpr(ible);
            if (expr is IConditionExpression ice) return ConvertConditionExpr(ice);
            if (expr is ICastExpression icaste) return ConvertCastExpr(icaste);
            if (expr is ICheckedExpression iche) return ConvertCheckedExpr(iche);
            if (expr is IBaseReferenceExpression ibre) return ConvertBaseRef(ibre);
            if (expr is ITypeReferenceExpression itre)
            {
                return ConvertTypeRefExpr(itre);
            }
            if (expr is ITypeOfExpression itoe)
            {
                return ConvertTypeOfExpr(itoe);
            }
            if (expr is IDelegateCreateExpression idce)
            {
                return ConvertDelegateCreate(idce);
            }
            if (expr is IObjectCreateExpression ioce)
            {
                return ConvertObjectCreate(ioce);
            }
            if (expr is IPropertyIndexerExpression ipie) return ConvertPropertyIndexerExpr(ipie);
            if (expr is IAddressDereferenceExpression iade) return ConvertAddressDereference(iade);
            if (expr is IAddressOutExpression iaoe) return ConvertAddressOut(iaoe);
            if (expr is IAnonymousMethodExpression iame) return ConvertAnonymousMethodExpression(iame);
            if (expr is IDefaultExpression ide) return ConvertDefaultExpr(ide);
            if (expr is IEventReferenceExpression iere) return ConvertEventRefExpr(iere);
            if (expr is IDelegateInvokeExpression idie) return ConvertDelegateInvoke(idie);
            if (expr is ILambdaExpression lambda) return ConvertLambda(lambda);
            if (expr == null) Error("expression is null");
            else Error("Unhandled expression: " + expr + " " + expr.GetType().Name);
            return expr;
        }

        private IExpression ConvertLambda(ILambdaExpression iLambdaExpression)
        {
            ILambdaExpression ile = new CodeModel.Concrete.XLambdaExpression();
            ile.Body = ConvertExpression(iLambdaExpression.Body);
            foreach (var ivd in iLambdaExpression.Parameters)
            {
                ile.Parameters.Add(ConvertVariableDecl(ivd));
            }
            return ile;
        }

        /// <summary>
        /// Shallow copy of BlockStatement
        /// </summary>
        /// <param name="inputBlock"></param>
        /// <returns></returns>
        protected virtual IBlockStatement ConvertBlockAlreadyOpen(IBlockStatement inputBlock)
        {
            bool wasCopy = ShallowCopy;
            ShallowCopy = false;
            IBlockStatement outputBlock = Builder.BlockStmt();
            context.SetPrimaryOutput(outputBlock);
            ConvertStatements(outputBlock.Statements, inputBlock.Statements);
            ShallowCopy = wasCopy;
            if (!ShallowCopy && ItemsAreReferenceEqual(outputBlock.Statements, inputBlock.Statements))
            {
                return inputBlock;
            }
            else
            {
                context.InputAttributes.CopyObjectAttributesTo(inputBlock, context.OutputAttributes, outputBlock);
            }
            return outputBlock;
        }

        protected virtual IStatement ConvertCondition(IConditionStatement ics)
        {
            IConditionStatement cs = Builder.CondStmt();
            context.SetPrimaryOutput(cs);
            bool wasCopy = ShallowCopy;
            ShallowCopy = false;
            cs.Condition = ConvertExpression(ics.Condition);
            ShallowCopy = wasCopy;
            cs.Then = ConvertBlock(ics.Then);
            if (ics.Else != null) cs.Else = ConvertBlock(ics.Else);
            if (cs.Then.Statements.Count == 0 && (cs.Else == null || cs.Else.Statements.Count == 0)) return null;
            if (!ShallowCopy && ReferenceEquals(cs.Condition, ics.Condition) && ReferenceEquals(cs.Then, ics.Then) && ReferenceEquals(cs.Else, ics.Else))
                return ics;
            context.InputAttributes.CopyObjectAttributesTo(ics, context.OutputAttributes, cs);
            return cs;
        }

        protected virtual IStatement ConvertExpressionStatement(IExpressionStatement ies)
        {
            IExpressionStatement es = Builder.ExprStatement();
            bool wasCopy = ShallowCopy;
            ShallowCopy = false;
            es.Expression = ConvertExpression(ies.Expression);
            ShallowCopy = wasCopy;
            if (es.Expression == null) return null;
            if (!ShallowCopy && ReferenceEquals(es.Expression, ies.Expression)) return ies;
            context.InputAttributes.CopyObjectAttributesTo(ies, context.OutputAttributes, es);
            return es;
        }

        protected virtual IStatement ConvertFor(IForStatement ifs)
        {
            IForStatement fs = Builder.ForStmt(ifs);
            context.SetPrimaryOutput(fs);
            bool wasCopy = ShallowCopy;
            ShallowCopy = false;
            fs.Initializer = ConvertStatement(ifs.Initializer);
            fs.Condition = ConvertExpression(ifs.Condition);
            fs.Increment = ConvertStatement(ifs.Increment);
            ShallowCopy = wasCopy;
            fs.Body = ConvertBlock(ifs.Body);
            if (fs.Body.Statements.Count == 0) return null;
            if (!ShallowCopy &&
                ReferenceEquals(fs.Body, ifs.Body) &&
                ReferenceEquals(fs.Initializer, ifs.Initializer) &&
                ReferenceEquals(fs.Condition, ifs.Condition) &&
                ReferenceEquals(fs.Increment, ifs.Increment)) return ifs;
            context.InputAttributes.CopyObjectAttributesTo(ifs, context.OutputAttributes, fs);
            return fs;
        }

        protected virtual IStatement ConvertRepeat(IRepeatStatement irs)
        {
            IRepeatStatement rs = Builder.RepeatStmt();
            context.SetPrimaryOutput(rs);
            bool wasCopy = ShallowCopy;
            ShallowCopy = false;
            rs.Count = ConvertExpression(irs.Count);
            ShallowCopy = wasCopy;
            rs.Body = ConvertBlock(irs.Body);
            if (rs.Body.Statements.Count == 0) return null;
            if (!ShallowCopy &&
                ReferenceEquals(rs.Body, irs.Body) &&
                ReferenceEquals(rs.Count, irs.Count)) return irs;
            context.InputAttributes.CopyObjectAttributesTo(irs, context.OutputAttributes, rs);
            return rs;
        }

        protected virtual IStatement ConvertReturnStatement(IMethodReturnStatement imrs)
        {
            if (imrs.Expression == null)
            {
                if (ShallowCopy) return Builder.Return();
                else return imrs;
            }
            bool wasCopy = ShallowCopy;
            ShallowCopy = false;
            IExpression newExpr = ConvertExpression(imrs.Expression);
            ShallowCopy = wasCopy;
            if (!ShallowCopy && ReferenceEquals(newExpr, imrs.Expression)) return imrs;
            return Builder.Return(newExpr);
        }

        protected virtual IStatement ConvertComment(ICommentStatement ics)
        {
            if (ShallowCopy)
            {
                ICommentStatement cs = Builder.CommentStmt(ics.Comment.Text);
                context.InputAttributes.CopyObjectAttributesTo(ics, context.OutputAttributes, cs);
                return cs;
            }
            else return ics;
        }

        private bool ItemsAreReferenceEqual<T>(IList<T> sc, IList<T> sc2)
        {
            if (sc.Count != sc2.Count) return false;
            for (int i = 0; i < sc.Count; i++)
            {
                if (!ReferenceEquals(sc[i], sc2[i])) return false;
            }
            return true;
        }

        protected virtual IExpression ConvertDelegateInvoke(IDelegateInvokeExpression imie)
        {
            IDelegateInvokeExpression mie = Builder.DelegateInvokeExpr();
            ConvertCollection(mie.Arguments, imie.Arguments);
            mie.Target = ConvertExpression(imie.Target);
            if (ItemsAreReferenceEqual(mie.Arguments, imie.Arguments) && ReferenceEquals(mie.Target, imie.Target)) return imie;
            context.InputAttributes.CopyObjectAttributesTo(imie, context.OutputAttributes, mie);
            return mie;
        }

        protected virtual IExpression ConvertAnonymousMethodExpression(IAnonymousMethodExpression iame)
        {
            IBlockStatement newBody = ConvertBlock(iame.Body);
            if (ReferenceEquals(newBody, iame.Body)) return iame;
            IAnonymousMethodExpression ame = Builder.AnonMethodExpr();
            ame.Body = newBody;
            ame.DelegateType = iame.DelegateType;
            for (int i = 0; i < iame.Parameters.Count; i++)
            {
                IParameterDeclaration pd = ConvertMethodParameter(iame.Parameters[i], i);
                ame.Parameters.Add(pd);
            }
            return ame;
        }

        protected virtual IExpression ConvertDefaultExpr(IDefaultExpression ide)
        {
            if (ide.Type is ITypeReference itr) ConvertTypeReference(itr);
            return ide;
        }

        protected virtual IExpression ConvertAddressOut(IAddressOutExpression iaoe)
        {
            IExpression newExpr = ConvertExpression(iaoe.Expression);
            if (ReferenceEquals(newExpr, iaoe.Expression))
                return iaoe;
            IAddressOutExpression aoe = Builder.AddrOutExpr();
            aoe.Expression = newExpr;
            return aoe;
        }

        protected virtual IExpression ConvertAddressDereference(IAddressDereferenceExpression iade)
        {
            ConvertExpression(iade.Expression);
            return iade;
        }

        protected virtual IExpression ConvertPropertyIndexerExpr(IPropertyIndexerExpression ipie)
        {
            IList<IExpression> newIndices = ConvertCollection(ipie.Indices);
            IPropertyReferenceExpression newTarget = (IPropertyReferenceExpression)ConvertExpression(ipie.Target);
            if (ReferenceEquals(newIndices, ipie.Indices) && ReferenceEquals(newTarget, ipie.Target))
                return ipie;
            IPropertyIndexerExpression pie = Builder.PropIndxrExpr();
            pie.Indices.AddRange(newIndices);
            pie.Target = newTarget;
            return ipie;
        }

        protected virtual IExpression ConvertBaseRef(IBaseReferenceExpression ibre)
        {
            return ibre;
        }

        protected virtual IExpression ConvertCastExpr(ICastExpression ice)
        {
            IExpression newExpr = ConvertExpression(ice.Expression);
            if (ReferenceEquals(newExpr, ice.Expression))
                return ice;
            ICastExpression ce = Builder.CastExpr(newExpr, ice.TargetType);
            return ce;
        }

        protected virtual IExpression ConvertCheckedExpr(ICheckedExpression ice)
        {
            IExpression newExpr = ConvertExpression(ice.Expression);
            if (ReferenceEquals(newExpr, ice.Expression))
                return ice;
            return Builder.CheckedExpr(newExpr);
        }

        protected virtual IExpression ConvertConditionExpr(IConditionExpression ice)
        {
            IExpression newCondition = ConvertExpression(ice.Condition);
            IExpression newThen = ConvertExpression(ice.Then);
            IExpression newElse = ConvertExpression(ice.Else);
            if (ReferenceEquals(newCondition, ice.Condition) && ReferenceEquals(newThen, ice.Then) && ReferenceEquals(newElse, ice.Else))
                return ice;
            IConditionExpression ce = Builder.CondExpr();
            ce.Condition = newCondition;
            ce.Then = newThen;
            ce.Else = newElse;
            return ce;
        }

        protected virtual IExpression ConvertBlockExpr(IBlockExpression ibe)
        {
            IList<IExpression> newExprs = ConvertCollection(ibe.Expressions);
            if (ReferenceEquals(newExprs, ibe.Expressions))
                return ibe;
            IBlockExpression be = Builder.BlockExpr();
            be.Expressions.AddRange(newExprs);
            return be;
        }


        protected virtual IExpression ConvertTypeRefExpr(ITypeReferenceExpression itre)
        {
            ConvertTypeReference(itre.Type);
            return itre;
        }

        protected virtual ITypeReference ConvertTypeReference(ITypeReference itr)
        {
            return itr;
        }

        protected virtual IExpression ConvertFieldRefExpr(IFieldReferenceExpression ifre)
        {
            IExpression newTarget = ConvertExpression(ifre.Target);
            if (ReferenceEquals(newTarget, ifre.Target)) return ifre;
            IFieldReferenceExpression fre = Builder.FieldRefExpr();
            fre.Target = newTarget;
            fre.Field = ifre.Field;
            return fre;
        }

        protected virtual IExpression ConvertPropertyRefExpr(IPropertyReferenceExpression ipre)
        {
            IExpression newTarget = ConvertExpression(ipre.Target);
            if (ReferenceEquals(newTarget, ipre.Target)) return ipre;
            IPropertyReferenceExpression pre = Builder.PropRefExpr();
            pre.Target = newTarget;
            pre.Property = ipre.Property;
            return pre;
        }

        protected virtual IExpression ConvertEventRefExpr(IEventReferenceExpression iere)
        {
            IExpression newTarget = ConvertExpression(iere.Target);
            if (ReferenceEquals(newTarget, iere.Target)) return iere;
            IEventReferenceExpression ere = Builder.EventRefExpr();
            ere.Target = newTarget;
            ere.Event = iere.Event;
            context.InputAttributes.CopyObjectAttributesTo(iere, context.OutputAttributes, ere);
            return ere;
        }

        protected virtual IExpression ConvertVariableDeclExpr(IVariableDeclarationExpression ivde)
        {
            IVariableDeclaration vd = ConvertVariableDecl(ivde.Variable);
            if (ReferenceEquals(vd, ivde.Variable)) return ivde;
            IVariableDeclarationExpression vde = Builder.VarDeclExpr(vd);
            context.InputAttributes.CopyObjectAttributesTo(vde, context.OutputAttributes, ivde);
            return vde;
        }

        protected virtual IVariableDeclaration ConvertVariableDecl(IVariableDeclaration ivd)
        {
            return ivd;
        }

        protected virtual IExpression ConvertVariableRefExpr(IVariableReferenceExpression ivre)
        {
            return ivre;
        }

        protected virtual IExpression ConvertArrayCreate(IArrayCreateExpression iace)
        {
            IList<IExpression> newDims = ConvertCollection(iace.Dimensions);
            IBlockExpression newInitializer = (iace.Initializer == null) ? null : (IBlockExpression)ConvertBlockExpr(iace.Initializer);
            if (ReferenceEquals(newDims, iace.Dimensions) && ReferenceEquals(newInitializer, iace.Initializer))
                return iace;
            IArrayCreateExpression ace = Builder.ArrayCreateExpr();
            ace.Dimensions.AddRange(newDims);
            ace.Initializer = newInitializer;
            ace.Type = iace.Type;
            return ace;
        }

        protected virtual IExpression ConvertArgumentRef(IArgumentReferenceExpression iare)
        {
            return iare;
        }

        protected virtual IExpression ConvertObjectCreate(IObjectCreateExpression ioce)
        {
            IList<IExpression> newArgs = ConvertCollection(ioce.Arguments);
            IBlockExpression newInitializer = (ioce.Initializer == null) ? null : (IBlockExpression)ConvertBlockExpr(ioce.Initializer);
            IMethodReference newConstructor = ConvertMethodReference(ioce.Constructor);
            if (ReferenceEquals(newArgs, ioce.Arguments) &&
                ReferenceEquals(newConstructor, ioce.Constructor) &&
                ReferenceEquals(newInitializer, ioce.Initializer))
                return ioce;
            IObjectCreateExpression oce = Builder.ObjCreateExpr();
            oce.Arguments.AddRange(newArgs);
            oce.Constructor = newConstructor;
            oce.Initializer = newInitializer;
            oce.Type = ioce.Type;
            return oce;
        }

        protected virtual IExpression ConvertArrayIndexer(IArrayIndexerExpression iaie)
        {
            IList<IExpression> newIndices = ConvertCollection(iaie.Indices);
            IExpression newTarget = ConvertExpression(iaie.Target);
            if (ReferenceEquals(newTarget, iaie.Target) &&
                ReferenceEquals(newIndices, iaie.Indices))
                return iaie;
            IArrayIndexerExpression aie = Builder.ArrayIndxrExpr();
            aie.Target = newTarget;
            aie.Indices.AddRange(newIndices);
            Context.InputAttributes.CopyObjectAttributesTo(iaie, context.OutputAttributes, aie);
            return aie;
        }

        protected virtual IExpression ConvertUnary(IUnaryExpression iue)
        {
            IExpression newExpr = ConvertExpression(iue.Expression);
            if (ReferenceEquals(newExpr, iue.Expression))
                return iue;
            IUnaryExpression ue = Builder.UnaryExpr(iue.Operator, newExpr);
            return ue;
        }

        protected virtual IExpression ConvertBinary(IBinaryExpression ibe)
        {
            IExpression newLeft = ConvertExpression(ibe.Left);
            IExpression newRight = ConvertExpression(ibe.Right);
            if (ReferenceEquals(newLeft, ibe.Left) && ReferenceEquals(newRight, ibe.Right))
                return ibe;
            IBinaryExpression be = Builder.BinaryExpr(newLeft, ibe.Operator, newRight);
            return be;
        }

        protected virtual IExpression ConvertLiteral(ILiteralExpression ile)
        {
            return ile;
        }

        protected virtual IExpression ConvertThis(IThisReferenceExpression itre)
        {
            return itre;
        }

        protected virtual IExpression ConvertAssign(IAssignExpression iae)
        {
            IExpression expr = ConvertExpression(iae.Expression);
            IExpression target = ConvertExpression(iae.Target);
            if (ReferenceEquals(expr, iae.Expression) && ReferenceEquals(target, iae.Target)) return iae;
            IAssignExpression ae = Builder.AssignExpr();
            ae.Expression = expr;
            ae.Target = target;
            context.InputAttributes.CopyObjectAttributesTo(iae, context.OutputAttributes, ae);
            return ae;
        }

        protected virtual IList<IExpression> ConvertCollection(IList<IExpression> exprColl)
        {
            IList<IExpression> newExprs = null;
            int count = 0;
            foreach (IExpression arg in exprColl)
            {
                IExpression newArg = ConvertExpression(arg);
                if (newExprs != null) newExprs.Add(newArg);
                else if (!ReferenceEquals(newArg, arg))
                {
                    newExprs = Builder.ExprCollection();
                    for (int i = 0; i < count; i++)
                    {
                        newExprs.Add(exprColl[i]);
                    }
                    newExprs.Add(newArg);
                }
                count++;
            }
            return newExprs ?? exprColl;
        }

        protected virtual IExpression ConvertMethodInvoke(IMethodInvokeExpression imie)
        {
            bool wasCopy = ShallowCopy;
            ShallowCopy = false;
            IList<IExpression> newArgs = ConvertCollection(imie.Arguments);
            IExpression newMethod = ConvertExpression(imie.Method);
            ShallowCopy = wasCopy;
            if (!ShallowCopy && ReferenceEquals(newArgs, imie.Arguments) && ReferenceEquals(newMethod, imie.Method))
                return imie;
            IMethodInvokeExpression mie = Builder.MethodInvkExpr();
            mie.Method = (IMethodReferenceExpression)newMethod;
            mie.Arguments.AddRange(newArgs);
            context.InputAttributes.CopyObjectAttributesTo(imie, context.OutputAttributes, mie);
            return mie;
        }

        protected virtual IExpression ConvertMethodRefExpr(IMethodReferenceExpression imre)
        {
            IExpression newTarget = ConvertExpression(imre.Target);
            IMethodReference newMethod = ConvertMethodReference(imre.Method);
            if (ReferenceEquals(newTarget, imre.Target) && ReferenceEquals(newMethod, imre.Method))
                return imre;
            return Builder.MethodRefExpr(newMethod, newTarget);
        }

        protected virtual IExpression ConvertTypeOfExpr(ITypeOfExpression itoe)
        {
            return itoe;
        }

        protected virtual IMethodReference ConvertMethodReference(IMethodReference imr)
        {
            return imr;
        }

        private IExpression ConvertDelegateCreate(IDelegateCreateExpression idce)
        {
            return idce;
        }

        protected virtual void ConvertCollection(IList<IExpression> outputs, IList<IExpression> inputs)
        {
            foreach (IExpression arg in inputs)
            {
                IExpression expr = ConvertExpression(arg);
                outputs.Add(expr);
            }
        }

        protected virtual IMethodReference ConvertMethodInstanceReference(IMethodReference imr)
        {
            IMethodReference mir = Builder.MethodInstRef();
            foreach (IType it in imr.GenericArguments)
            {
                mir.GenericArguments.Add(it);
            }
            mir.MethodInfo = imr.MethodInfo;
            mir.GenericMethod = ConvertMethodReference(imr.GenericMethod);
            mir.DeclaringType = imr.DeclaringType; // TODO: INCORRECT - MUST BE CONVERTED!!
            return mir;
        }

        protected void Warning(string msg)
        {
            Context.Warning(msg);
        }

        protected void Error(string msg)
        {
            Context.Error(msg);
        }

        protected void Warning(string msg, Exception ex)
        {
            context.Warning(msg, ex);
        }

        protected void Error(string msg, Exception ex)
        {
            context.Error(msg, ex);
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}