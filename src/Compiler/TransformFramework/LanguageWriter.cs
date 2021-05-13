// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

// Language writer - converts type defined in code model into
// a source code tree
// This writer serves two purposes
// (a) To be able to write out code for compilation from the AST
// (b) To maintain a 1-1 relationship between AST and source fragments
//     so that the transform browser can maintain a mapping between
//     the source fragments of different transforms
// This writer is by no means exhaustive - many code constructs
// are not supported.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Reflection;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Collections;

namespace Microsoft.ML.Probabilistic.Compiler
{
    /// <summary>
    /// Language writer base class. It is intended that classes implementing
    /// ILanguageWriter should inherit from this class.
    /// The base class manages the general traversal, generating a tree of source nodes.
    /// Specific leaf details are filled in by derived classes
    /// </summary>
    internal abstract class LanguageWriter : ILanguageWriter
    {
        /// <summary>
        /// Helps build class declarations
        /// </summary>
        protected static readonly CodeBuilder Builder = CodeBuilder.Instance;

        /// <summary>
        /// node stack representing the current path in the tree node collection
        /// </summary>
        protected Stack<SourceNode> nodeStack = new Stack<SourceNode>();

        // Current tab
        protected int currTab;

        /// <summary>
        /// Dictionary mapping type reference names to namespace
        /// </summary>
        protected Dictionary<string, string> typeReferenceMap = new Dictionary<string, string>();

        private readonly Set<Assembly> referencedAssemblies = new Set<Assembly>();

        public ICollection<Assembly> ReferencedAssemblies
        {
            get { return referencedAssemblies; }
        }

        /// <summary>
        /// List of tabs
        /// </summary>
        protected List<string> tabStrings = new List<string>();

        /// <summary>
        /// When writing code inside a namespace declaration, the name of that namespace.  Otherwise null.
        /// </summary>
        protected string CurrentNamespace;

        /// <summary>
        /// Flag that tracks if there's a conflict in type names. If so,
        /// the specific type will need to be fully qualified.
        /// </summary>
        protected bool isConflict = false;

        /// <summary>
        /// Binary operator look-up table
        /// </summary>
        protected static readonly Dictionary<BinaryOperator, string> BinaryOpLookUp = new Dictionary<BinaryOperator, string>();

        /// <summary>
        /// Unary operator look-up table
        /// </summary>
        protected static readonly Dictionary<UnaryOperator, string> UnaryOpLookUp = new Dictionary<UnaryOperator, string>();

        /// <summary>
        /// Type look-up table for intrinsic dotNet types
        /// </summary>
        protected static readonly Dictionary<string, string> IntrinsicTypeAlias = new Dictionary<string, string>();

        protected static readonly Dictionary<string, string> ReservedMap = new Dictionary<string, string>();

        // Following all need to be overridden
        protected abstract void AppendInterfaces(StringBuilder sb, List<ITypeReference> interfaces, bool hasBaseType);
        protected abstract void AppendAttribute(StringBuilder sb, ICustomAttribute attr);
        protected abstract void AppendGenericArguments(StringBuilder sb, IEnumerable<IType> genericArguments);
        protected abstract void AppendArrayRank(StringBuilder sb, int ar);
        protected abstract void AppendVariableDeclaration(StringBuilder sb, IVariableDeclaration ivd);
        protected abstract void AppendAddressOutExpression(StringBuilder sb, IAddressOutExpression iae);
        protected abstract void AppendArgumentReferenceExpression(StringBuilder sb, IArgumentReferenceExpression iare);
        protected abstract void AppendArrayCreateExpression(StringBuilder sb, IArrayCreateExpression iae);
        protected abstract void AppendArrayIndexerExpression(StringBuilder sb, IArrayIndexerExpression iaie);
        protected abstract void AppendAssignExpression(StringBuilder sb, IAssignExpression iae);
        protected abstract void AppendBaseReferenceExpression(StringBuilder sb, IBaseReferenceExpression ibre);
        protected abstract void AppendBinaryExpression(StringBuilder sb, IBinaryExpression ibe);
        protected abstract void AppendBlockExpression(StringBuilder sb, IBlockExpression ibe);
        protected abstract void AppendCanCastExpression(StringBuilder sb, ICanCastExpression ice);
        protected abstract void AppendCastExpression(StringBuilder sb, ICastExpression ice);
        protected abstract void AppendCheckedExpression(StringBuilder sb, ICheckedExpression ice);
        protected abstract void AppendConditionExpression(StringBuilder sb, IConditionExpression ice);
        protected abstract void AppendDelegateCreateExpression(StringBuilder sb, IDelegateCreateExpression idce);
        protected abstract void AppendEventReferenceExpression(StringBuilder sb, IEventReferenceExpression iere);
        protected abstract void AppendFieldReferenceExpression(StringBuilder sb, IFieldReferenceExpression ifre);
        protected abstract void AppendLambdaExpression(StringBuilder sb, ILambdaExpression ile);
        protected abstract void AppendLiteralExpression(StringBuilder sb, ILiteralExpression ile);
        protected abstract void AppendMethodInvokeExpression(StringBuilder sb, IMethodInvokeExpression imie);
        protected abstract void AppendMethodReferenceExpression(StringBuilder sb, IMethodReferenceExpression imre);
        protected abstract void AppendObjectCreateExpression(StringBuilder sb, IObjectCreateExpression ioce);
        protected abstract void AppendPropertyIndexerExpression(StringBuilder sb, IPropertyIndexerExpression ipie);
        protected abstract void AppendPropertyReferenceExpression(StringBuilder sb, IPropertyReferenceExpression ipre);
        protected abstract void AppendThisReferenceExpression(StringBuilder sb, IThisReferenceExpression itre);
        protected abstract void AppendTypeOfExpression(StringBuilder sb, ITypeOfExpression itoe);
        protected abstract void AppendTypeReferenceExpression(StringBuilder sb, ITypeReferenceExpression itre);
        protected abstract void AppendUnaryExpression(StringBuilder sb, IUnaryExpression iue);
        protected abstract void AppendVariableDeclarationExpression(StringBuilder sb, IVariableDeclarationExpression ivde);
        protected abstract void AppendVariableReferenceExpression(StringBuilder sb, IVariableReferenceExpression ivre);
        protected abstract void AppendAddressReferenceExpression(StringBuilder sb, IAddressReferenceExpression iare);
        protected abstract void AppendAddressDereferenceExpression(StringBuilder sb, IAddressDereferenceExpression iade);
        protected abstract void AppendAnonymousMethodExpression(StringBuilder sb, IAnonymousMethodExpression iame);
        protected abstract void AppendMemberInitializerExpression(StringBuilder sb, IMemberInitializerExpression imie);
        protected abstract void AppendDelegateInvokeExpression(StringBuilder sb, IDelegateInvokeExpression idie);
        protected abstract void AppendDefaultExpression(StringBuilder sb, IDefaultExpression ide);
        protected abstract void AppendParameterDeclaration(StringBuilder sb, IParameterDeclaration parameters);
        protected abstract void AppendParameterDeclarationCollection(StringBuilder sb, IList<IParameterDeclaration> parameters);
        protected abstract void AttachPropertyAccessor(IMethodReference imr, bool set);
        public abstract SourceNode GenerateSource(ITypeDeclaration itd);
        public abstract SourceNode GeneratePartialSource(ITypeDeclaration itd);
        public abstract SourceNode GeneratePartialSource(IStatement ist);
        public abstract SourceNode AttachTypeDeclaration(ITypeDeclaration itd);

        // Following return a string of length 0. Most of these will need to be overridden
        protected virtual string BlockStatementStart(IBlockStatement ibs)
        {
            return "";
        }

        protected virtual string BlockStatementEnd(IBlockStatement ibs)
        {
            return "";
        }

        protected virtual string BreakStatementStart(IBreakStatement ibs)
        {
            return "";
        }

        protected virtual string BreakStatementEnd(IBreakStatement ibs)
        {
            return "";
        }

        protected virtual string CommentStatementStart(ICommentStatement ics)
        {
            return "";
        }

        protected virtual string CommentStatementEnd(ICommentStatement ics)
        {
            return "";
        }

        protected virtual string IfStatementStart(IConditionStatement ics)
        {
            return "";
        }

        protected virtual string IfStatementEnd(IConditionStatement ics)
        {
            return "";
        }

        protected virtual string ElseStatementStart(IConditionStatement ics)
        {
            return "";
        }

        protected virtual string ElseStatementEnd(IConditionStatement ics)
        {
            return "";
        }

        protected virtual string ContinueStatementStart(IContinueStatement ics)
        {
            return "";
        }

        protected virtual string ContinueStatementEnd(IContinueStatement ics)
        {
            return "";
        }

        protected virtual string ExpressionStatementStart(IExpressionStatement ies)
        {
            return "";
        }

        protected virtual string ExpressionStatementEnd(IExpressionStatement ies)
        {
            return "";
        }

        protected virtual string ForEachStatementStart(IForEachStatement ifes)
        {
            return "";
        }

        protected virtual string ForEachStatementEnd(IForEachStatement ifes)
        {
            return "";
        }

        protected virtual string ForStatementStart(IForStatement ifs)
        {
            return "";
        }

        protected virtual string ForStatementEnd(IForStatement ifs)
        {
            return "";
        }

        protected virtual string RepeatStatementStart(IRepeatStatement ifs)
        {
            return "";
        }

        protected virtual string RepeatStatementEnd(IRepeatStatement ifs)
        {
            return "";
        }

        protected virtual string MethodReturnStatementStart(IMethodReturnStatement imrs)
        {
            return "";
        }

        protected virtual string MethodReturnStatementEnd(IMethodReturnStatement imrs)
        {
            return "";
        }

        protected virtual string SwitchStatementStart(ISwitchStatement iss)
        {
            return "";
        }

        protected virtual string SwitchStatementEnd(ISwitchStatement iss)
        {
            return "";
        }

        protected virtual string SwitchCaseStart(ISwitchCase isc)
        {
            return "";
        }

        protected virtual string SwitchCaseEnd(ISwitchCase isc)
        {
            return "";
        }

        protected virtual string ThrowExceptionStart(IThrowExceptionStatement ites)
        {
            return "";
        }

        protected virtual string ThrowExceptionEnd(IThrowExceptionStatement ites)
        {
            return "";
        }

        protected virtual string TryBlockStart(ITryCatchFinallyStatement itcfs)
        {
            return "";
        }

        protected virtual string TryBlockEnd(ITryCatchFinallyStatement itcfs)
        {
            return "";
        }

        protected virtual string FaultBlockStart(ITryCatchFinallyStatement itcfs)
        {
            return "";
        }

        protected virtual string FaultBlockEnd(ITryCatchFinallyStatement itcfs)
        {
            return "";
        }

        protected virtual string CatchClauseStart(ICatchClause itcfs)
        {
            return "";
        }

        protected virtual string CatchClauseEnd(ICatchClause itcfs)
        {
            return "";
        }

        protected virtual string FinallyBlockStart(ITryCatchFinallyStatement itcfs)
        {
            return "";
        }

        protected virtual string FinallyBlockEnd(ITryCatchFinallyStatement itcfs)
        {
            return "";
        }

        protected virtual string UsingStatementStart(IUsingStatement ius)
        {
            return "";
        }

        protected virtual string UsingStatementEnd(IUsingStatement ius)
        {
            return "";
        }

        protected virtual string WhileStatementStart(IWhileStatement iws)
        {
            return "";
        }

        protected virtual string WhileStatementEnd(IWhileStatement iws)
        {
            return "";
        }

        protected virtual string MethodDeclarationStart(IMethodDeclaration imd)
        {
            return "";
        }

        protected virtual string MethodDeclarationEnd(IMethodDeclaration imd)
        {
            return "";
        }

        protected virtual string FieldDeclarationStart(IFieldDeclaration ifd)
        {
            return "";
        }

        protected virtual string FieldDeclarationEnd(IFieldDeclaration ifd)
        {
            return "";
        }

        protected virtual string PropertyDeclarationCollectionStart(List<IPropertyDeclaration> ipdc)
        {
            return "";
        }

        protected virtual string PropertyDeclarationCollectionEnd(List<IPropertyDeclaration> ipdc)
        {
            return "";
        }

        protected virtual string PropertyDeclarationStart(IPropertyDeclaration ipd)
        {
            return "";
        }

        protected virtual string PropertyDeclarationEnd(IPropertyDeclaration ipd)
        {
            return "";
        }

        protected virtual string EventDeclarationCollectionStart(List<IEventDeclaration> iedc)
        {
            return "";
        }

        protected virtual string EventDeclarationCollectionEnd(List<IEventDeclaration> iedc)
        {
            return "";
        }

        protected virtual string EventDeclarationStart(IEventDeclaration ied)
        {
            return "";
        }

        protected virtual string EventDeclarationEnd(IEventDeclaration ied)
        {
            return "";
        }

        protected virtual string TypeDeclarationCollectionStart(List<ITypeDeclaration> intdc)
        {
            return "";
        }

        protected virtual string TypeDeclarationCollectionEnd(List<ITypeDeclaration> intdc)
        {
            return "";
        }

        protected virtual string FieldDeclarationCollectionStart(List<IFieldDeclaration> ifdc)
        {
            return "";
        }

        protected virtual string FieldDeclarationCollectionEnd(List<IFieldDeclaration> ifdc)
        {
            return "";
        }

        protected virtual string MethodDeclarationCollectionStart(List<IMethodDeclaration> imdc)
        {
            return "";
        }

        protected virtual string MethodDeclarationCollectionEnd(List<IMethodDeclaration> imdc)
        {
            return "";
        }

        /// <summary>
        /// Initialise the language writer
        /// </summary>
        public virtual void Initialise()
        {
            nodeStack.Clear();
            typeReferenceMap.Clear();
            tabStrings.Clear();
            currTab = 0;

            isConflict = false;
        }

        /// <summary>
        /// Return a valid identifier - this deals with reserved words in the
        /// particular language
        /// </summary>
        /// <param name="ident">Identifier</param>
        /// <returns>A valid identifier</returns>
        protected string ValidIdentifier(string ident)
        {
            if (ident != null && ReservedMap.ContainsKey(ident))
                return ReservedMap[ident];
            else
                return ident;
        }

        /// <summary>
        /// Get the current tab string
        /// </summary>
        /// <returns></returns>
        protected virtual string GetTabString()
        {
            int cnt = tabStrings.Count;
            if (cnt <= currTab)
            {
                string curr = "";
                if (cnt > 0)
                    curr = tabStrings[cnt - 1] + "\t";
                for (int i = cnt; i <= currTab; i++)
                {
                    tabStrings.Add(curr);
                    curr += "\t";
                }
            }
            return tabStrings[currTab];
        }

        /// <summary>
        /// Add a child to the current source node
        /// </summary>
        /// <param name="sn"></param>
        protected virtual void AddChild(SourceNode sn)
        {
            SourceNode parent = nodeStack.Peek();
            if (parent.Children == null)
                parent.Children = new List<SourceNode>();
            parent.Children.Add(sn);
        }

        /// <summary>
        /// Append an array type to a string builder
        /// </summary>
        /// <param name="sb"></param>
        /// <param name="iat"></param>
        protected virtual void AppendArrayType(StringBuilder sb, IArrayType iat)
        {
            AppendType(sb, iat.ElementType);
            AppendArrayRank(sb, iat.Rank);
        }


        /// <summary>
        /// Append a type reference to a string builder
        /// </summary>
        /// <param name="sb"></param>
        /// <param name="itr"></param>
        protected virtual void AppendTypeReference(StringBuilder sb, ITypeReference itr)
        {
            string typName = itr.Name;

            Type typ = itr.DotNetType;
            int usedGenericParamsCount = 0;
            var originalType = typ;
            if (typ != null)
            {
                StringBuilder sb2 = new StringBuilder();
                if (typ.IsNested && !typ.IsGenericParameter)
                {
                    typ = typ.DeclaringType;
                    // if typ contains generic parameters then we need to consume them from the original type
                    if (typ.ContainsGenericParameters)
                    {
                        usedGenericParamsCount = typ.GetGenericArguments().Length;
                        typ = typ.MakeGenericType(originalType.GetGenericArguments().Take(usedGenericParamsCount).ToArray());
                    }
                    AppendType(sb2, typ);
                    sb2.Append(".");
                }
                typName = sb2.ToString() + typName;
            }
            if (typ != null && IntrinsicTypeAlias.ContainsKey(typ.FullName))
            {
                typName = IntrinsicTypeAlias[typ.FullName];
            }
            else
            {
                if (CurrentNamespace != null)
                {
                    // Check for namespace conflicts.
                    bool hasConflict = GetNamespacePrefixes(CurrentNamespace).Any(parentNamespace => IsNamespace(parentNamespace, typName));
                    if (hasConflict) typName = typ.FullName;
                }
                if (typeReferenceMap.ContainsKey(typName))
                {
                    if (typeReferenceMap[typName] != itr.Namespace)
                        isConflict = true;
                }
                else
                {
                    if (itr.Namespace != "")
                        typeReferenceMap.Add(typName, itr.Namespace);
                }
            }
            AddReferencedAssembly(itr);
            // TODO: if two namespaces have the same type name, we need to use the full name to make sure we refer to the right namespace.
            sb.Append(typName);
            if (Builder.IsTypeInstRef(itr))
                AppendGenericArguments(sb, itr.GenericArguments.Skip(usedGenericParamsCount));
        }

        private HashSet<string> LoadedNamespaces;

        private bool IsNamespace(string parentNamespace, string name)
        {
            if(LoadedNamespaces == null)
            {
                LoadedNamespaces = new HashSet<string>(AppDomain.CurrentDomain.GetAssemblies()
                    .SelectMany(assembly => GetTypes(assembly))
                    .Select(type => type.Namespace ?? "")
                    .SelectMany(GetNamespacePrefixes));
            }
            return LoadedNamespaces.Contains(parentNamespace + "." + name);
        }

        private IEnumerable<Type> GetTypes(Assembly assembly)
        {
            try
            {
                return assembly.GetTypes();
            }
            catch
            {
                return new Type[0];
            }
        }

        private IEnumerable<string> GetNamespacePrefixes(string namespaceName)
        {
            yield return namespaceName;
            for (int i = namespaceName.Length - 1; i >= 0; i--)
            {
                if (namespaceName[i] == '.') yield return namespaceName.Substring(0, i);
            }
        }

        protected void AddReferencedAssembly(ITypeReference itr)
        {
            // find the parent Assembly and add a reference to it
            object owner = itr.Owner;
            while (owner is ITypeReference reference) owner = reference.Owner;
            if (owner is Assembly assembly) ReferencedAssemblies.Add(assembly);
            else if(owner != null) throw new InferCompilerException("unknown assembly for type reference: " + itr.DotNetType);
        }

        /// <summary>
        /// Append a reference type to a string builder
        /// </summary>
        /// <param name="sb"></param>
        /// <param name="irt"></param>
        protected virtual void AppendReferenceType(StringBuilder sb, IReferenceType irt)
        {
            sb.Append("out ");
            AppendType(sb, irt.ElementType);
        }

        /// <summary>
        /// Append a generic parameter type to a string builder
        /// </summary>
        /// <param name="sb"></param>
        /// <param name="igpt"></param>
        protected virtual void AppendGenericParameterType(StringBuilder sb, IGenericParameter igpt)
        {
            //AppendType(sb, igpt.Resolve());
            sb.Append(igpt.Name);
        }

        /// <summary>
        /// Append a generic argument type to a string builder
        /// </summary>
        /// <param name="sb"></param>
        /// <param name="igat"></param>
        protected virtual void AppendGenericArgumentType(StringBuilder sb, IGenericArgument igat)
        {
            AppendType(sb, igat.Resolve());
        }

        /// <summary>
        /// Append a type to a StringBuilder
        /// </summary>
        /// <param name="sb">The StringBuilder</param>
        /// <param name="it">The type</param>
        protected void AppendType(StringBuilder sb, Type it)
        {
            AppendType(sb, Builder.TypeRef(it));
        }

        /// <summary>
        /// Append a type to a StringBuilder
        /// </summary>
        /// <param name="sb">The StringBuilder</param>
        /// <param name="it">The type</param>
        protected virtual void AppendType(StringBuilder sb, IType it)
        {
            if (it == null)
                return;
            if (it is IArrayType) AppendArrayType(sb, it as IArrayType);
            else if (it is ITypeReference) AppendTypeReference(sb, it as ITypeReference);
            else if (it is IReferenceType) AppendReferenceType(sb, it as IReferenceType);
            else if (it is IGenericParameter) AppendGenericParameterType(sb, it as IGenericParameter);
            else if (it is IGenericArgument) AppendGenericArgumentType(sb, it as IGenericArgument);
            else throw new NotSupportedException("Language writer: unsupported type");
        }

        /// <summary>
        /// Type source
        /// </summary>
        /// <param name="it">Type</param>
        /// <returns>type source string</returns>
        public virtual string TypeSource(IType it)
        {
            Initialise();
            SourceNode root = new SourceNode();
            nodeStack.Push(root);
            StringBuilder sb = new StringBuilder();
            AppendType(sb, it);
            return sb.ToString();
        }

        /// <summary>
        /// Variable declaration source
        /// </summary>
        /// <param name="ivd">Variable declaration</param>
        /// <returns></returns>
        public virtual string VariableDeclarationSource(IVariableDeclaration ivd)
        {
            Initialise();
            SourceNode root = new SourceNode();
            nodeStack.Push(root);
            StringBuilder sb = new StringBuilder();
            AppendVariableDeclaration(sb, ivd);
            return sb.ToString();
        }

        /// <summary>
        /// Append an expression to a StringBuilder
        /// </summary>
        /// <param name="sb">The StringBuilder</param>
        /// <param name="ie">The IExpression</param>
        /// <remarks>This is not exhaustive</remarks>
        public virtual void AppendExpression(StringBuilder sb, IExpression ie)
        {
            if (ie == null)
                return;

            if (ie is IAddressOutExpression) AppendAddressOutExpression(sb, ie as IAddressOutExpression);
            else if (ie is IArgumentReferenceExpression) AppendArgumentReferenceExpression(sb, ie as IArgumentReferenceExpression);
            else if (ie is IArrayCreateExpression) AppendArrayCreateExpression(sb, ie as IArrayCreateExpression);
            else if (ie is IArrayIndexerExpression) AppendArrayIndexerExpression(sb, ie as IArrayIndexerExpression);
            else if (ie is IAssignExpression) AppendAssignExpression(sb, ie as IAssignExpression);
            else if (ie is IBaseReferenceExpression) AppendBaseReferenceExpression(sb, ie as IBaseReferenceExpression);
            else if (ie is IBinaryExpression) AppendBinaryExpression(sb, ie as IBinaryExpression);
            else if (ie is IBlockExpression) AppendBlockExpression(sb, ie as IBlockExpression);
            else if (ie is ICanCastExpression) AppendCanCastExpression(sb, ie as ICanCastExpression);
            else if (ie is ICastExpression) AppendCastExpression(sb, ie as ICastExpression);
            else if (ie is ICheckedExpression) AppendCheckedExpression(sb, ie as ICheckedExpression);
            else if (ie is IConditionExpression) AppendConditionExpression(sb, ie as IConditionExpression);
            else if (ie is IDelegateCreateExpression) AppendDelegateCreateExpression(sb, ie as IDelegateCreateExpression);
            else if (ie is IEventReferenceExpression) AppendEventReferenceExpression(sb, ie as IEventReferenceExpression);
            else if (ie is IFieldReferenceExpression) AppendFieldReferenceExpression(sb, ie as IFieldReferenceExpression);
            else if (ie is ILambdaExpression) AppendLambdaExpression(sb, ie as ILambdaExpression);
            else if (ie is ILiteralExpression) AppendLiteralExpression(sb, ie as ILiteralExpression);
            else if (ie is IMethodInvokeExpression) AppendMethodInvokeExpression(sb, ie as IMethodInvokeExpression);
            else if (ie is IMethodReferenceExpression) AppendMethodReferenceExpression(sb, ie as IMethodReferenceExpression);
            else if (ie is IObjectCreateExpression) AppendObjectCreateExpression(sb, ie as IObjectCreateExpression);
            else if (ie is IPropertyIndexerExpression) AppendPropertyIndexerExpression(sb, ie as IPropertyIndexerExpression);
            else if (ie is IPropertyReferenceExpression) AppendPropertyReferenceExpression(sb, ie as IPropertyReferenceExpression);
            else if (ie is IThisReferenceExpression) AppendThisReferenceExpression(sb, ie as IThisReferenceExpression);
            else if (ie is ITypeOfExpression) AppendTypeOfExpression(sb, ie as ITypeOfExpression);
            else if (ie is ITypeReferenceExpression) AppendTypeReferenceExpression(sb, ie as ITypeReferenceExpression);
            else if (ie is IUnaryExpression) AppendUnaryExpression(sb, ie as IUnaryExpression);
            else if (ie is IVariableDeclarationExpression) AppendVariableDeclarationExpression(sb, ie as IVariableDeclarationExpression);
            else if (ie is IVariableReferenceExpression) AppendVariableReferenceExpression(sb, ie as IVariableReferenceExpression);
            else if (ie is IAddressReferenceExpression) AppendAddressReferenceExpression(sb, ie as IAddressReferenceExpression);
            else if (ie is IAddressDereferenceExpression) AppendAddressDereferenceExpression(sb, ie as IAddressDereferenceExpression);
            else if (ie is IAnonymousMethodExpression) AppendAnonymousMethodExpression(sb, ie as IAnonymousMethodExpression);
            else if (ie is IMemberInitializerExpression) AppendMemberInitializerExpression(sb, ie as IMemberInitializerExpression);
            else if (ie is IDelegateInvokeExpression) AppendDelegateInvokeExpression(sb, ie as IDelegateInvokeExpression);
            else if (ie is IDefaultExpression) AppendDefaultExpression(sb, ie as IDefaultExpression);
            else throw new NotImplementedException("Don't know how to write: " + ie);
        }

        /// <summary>
        /// Expression source
        /// </summary>
        /// <param name="ie">Expression</param>
        /// <returns>expression source string</returns>
        public string ExpressionSource(IExpression ie)
        {
            Initialise();
            SourceNode root = new SourceNode();
            nodeStack.Push(root);
            StringBuilder sb = new StringBuilder();
            AppendExpression(sb, ie);
            return sb.ToString();
        }

        /// <summary>
        /// Parameter declaration source
        /// </summary>
        /// <param name="ipd">Parameter declaration</param>
        /// <returns></returns>
        public string ParameterDeclarationSource(IParameterDeclaration ipd)
        {
            Initialise();
            SourceNode root = new SourceNode();
            nodeStack.Push(root);
            StringBuilder sb = new StringBuilder();
            AppendParameterDeclaration(sb, ipd);
            return sb.ToString();
        }

        /// <summary>
        /// Parameter collection source
        /// </summary>
        /// <param name="parameters"></param>
        /// <returns></returns>
        public string ParameterDeclarationCollectionSource(IList<IParameterDeclaration> parameters)
        {
            Initialise();
            SourceNode root = new SourceNode();
            nodeStack.Push(root);
            StringBuilder sb = new StringBuilder();
            AppendParameterDeclarationCollection(sb, parameters);
            return sb.ToString();
        }

        /// <summary>
        /// This converts an input type into source, calling recursively into any
        /// methods, fields, properties, and/or nested types.
        /// </summary>
        /// <param name="itd">The input type declaration</param>
        public string TypeDeclarationSource(ITypeDeclaration itd)
        {
            this.Initialise();
            SourceNode root = new SourceNode();
            nodeStack.Push(root);
            return this.AttachTypeDeclaration(itd).ToString(this);
        }

        /// <summary>
        /// Attach block statement
        /// </summary>
        /// <param name="ibs">Block statement</param>
        /// <returns>source node</returns>
        public virtual SourceNode AttachBlockStatement(IBlockStatement ibs)
        {
            SourceNode sn = new SourceNode
            {
                ASTElement = ibs,
                StartString = BlockStatementStart(ibs),
                EndString = BlockStatementEnd(ibs)
            };
            AddChild(sn);
            nodeStack.Push(sn);
            AttachStatements(ibs.Statements);
            nodeStack.Pop();
            return sn;
        }

        /// <summary>
        /// Attach break statement
        /// </summary>
        /// <param name="ibs">Break statement</param>
        /// <returns>source node</returns>
        public virtual SourceNode AttachBreakStatement(IBreakStatement ibs)
        {
            SourceNode sn = new SourceNode
            {
                ASTElement = ibs,
                StartString = BreakStatementStart(ibs),
                EndString = BreakStatementEnd(ibs)
            };
            AddChild(sn);
            return sn;
        }

        /// <summary>
        /// Attach comment statement
        /// </summary>
        /// <param name="ics">Comment statement</param>
        /// <returns>source node</returns>
        public virtual SourceNode AttachCommentStatement(ICommentStatement ics)
        {
            SourceNode sn = new SourceNode
            {
                ASTElement = ics,
                StartString = CommentStatementStart(ics),
                EndString = CommentStatementEnd(ics)
            };
            AddChild(sn);
            return sn;
        }

        /// <summary>
        /// Attach condition statement
        /// </summary>
        /// <param name="ics">Condition statement</param>
        /// <returns>source node</returns>
        public virtual SourceNode AttachConditionStatement(IConditionStatement ics)
        {
            SourceNode sn = new SourceNode();
            sn.ASTElement = ics;
            AddChild(sn);
            nodeStack.Push(sn);

            // The if/then part of the statement
            SourceNode snIfThen = new SourceNode(IfStatementStart(ics), IfStatementEnd(ics), ics);
            AddChild(snIfThen);
            nodeStack.Push(snIfThen);
            currTab++;
            AttachStatements(ics.Then.Statements);
            currTab--;
            nodeStack.Pop();

            // The else part of the statement
            if (ics.Else != null)
            {
                SourceNode snElse = new SourceNode(ElseStatementStart(ics), ElseStatementEnd(ics), ics);
                AddChild(snElse);
                nodeStack.Push(snElse);
                currTab++;
                AttachStatements(ics.Else.Statements);
                currTab--;
                nodeStack.Pop();
            }
            nodeStack.Pop();
            return sn;
        }

        public virtual SourceNode AttachContinueStatement(IContinueStatement ics)
        {
            SourceNode sn = new SourceNode
            {
                ASTElement = ics,
                StartString = ContinueStatementStart(ics),
                EndString = ContinueStatementEnd(ics)
            };
            AddChild(sn);
            return sn;
        }

        /// <summary>
        /// Attach expression statement
        /// </summary>
        /// <param name="ies">Expression statement</param>
        /// <returns>source node</returns>
        public virtual SourceNode AttachExpressionStatement(IExpressionStatement ies)
        {
            SourceNode sn = new SourceNode
            {
                ASTElement = ies,
                StartString = ExpressionStatementStart(ies),
                EndString = ExpressionStatementEnd(ies)
            };
            AddChild(sn);
            return sn;
        }

        /// <summary>
        /// Attach foreach statement
        /// </summary>
        /// <param name="ifes">Foreach statement</param>
        /// <returns>source node</returns>
        public virtual SourceNode AttachForEachStatement(IForEachStatement ifes)
        {
            SourceNode sn = new SourceNode
            {
                ASTElement = ifes,
                StartString = ForEachStatementStart(ifes),
                EndString = ForEachStatementEnd(ifes)
            };
            AddChild(sn);
            nodeStack.Push(sn);
            currTab++;
            AttachStatements(ifes.Body.Statements);
            currTab--;
            nodeStack.Pop();
            return sn;
        }

        /// <summary>
        /// Attach for statement
        /// </summary>
        /// <param name="ifs">For statement</param>
        /// <returns>source node</returns>
        public virtual SourceNode AttachForStatement(IForStatement ifs)
        {
            SourceNode sn = new SourceNode
            {
                ASTElement = ifs,
                StartString = ForStatementStart(ifs),
                EndString = ForStatementEnd(ifs)
            };
            AddChild(sn);
            nodeStack.Push(sn);
            currTab++;
            AttachStatements(ifs.Body.Statements);
            currTab--;
            nodeStack.Pop();
            return sn;
        }


        /// <summary>
        /// Attach repeat statement
        /// </summary>
        /// <param name="irs">Repeat statement</param>
        /// <returns>source node</returns>
        public virtual SourceNode AttachRepeatStatement(IRepeatStatement irs)
        {
            SourceNode sn = new SourceNode
            {
                ASTElement = irs,
                StartString = RepeatStatementStart(irs),
                EndString = RepeatStatementEnd(irs)
            };
            AddChild(sn);
            nodeStack.Push(sn);
            currTab++;
            AttachStatements(irs.Body.Statements);
            currTab--;
            nodeStack.Pop();
            return sn;
        }

        /// <summary>
        /// Attach method return statement
        /// </summary>
        /// <param name="imrs">Method return statement</param>
        /// <returns>source node</returns>
        public virtual SourceNode AttachMethodReturnStatement(IMethodReturnStatement imrs)
        {
            SourceNode sn = new SourceNode
            {
                ASTElement = imrs,
                StartString = MethodReturnStatementStart(imrs),
                EndString = MethodReturnStatementEnd(imrs)
            };
            AddChild(sn);
            return sn;
        }

        /// <summary>
        /// Attach switch statement
        /// </summary>
        /// <param name="iss">Switch statement</param>
        /// <returns>source node</returns>
        public virtual SourceNode AttachSwitchStatement(ISwitchStatement iss)
        {
            SourceNode sn = new SourceNode
            {
                ASTElement = iss,
                StartString = SwitchStatementStart(iss),
                EndString = SwitchStatementEnd(iss)
            };
            AddChild(sn);
            nodeStack.Push(sn);
            currTab++;
            foreach (ISwitchCase isc in iss.Cases)
            {
                AttachSwitchCase(isc);
            }
            currTab--;
            nodeStack.Pop();
            return sn;
        }

        /// <summary>
        /// Attach Switch case
        /// </summary>
        /// <param name="isc">Switch case</param>
        /// <returns>source node</returns>
        public virtual SourceNode AttachSwitchCase(ISwitchCase isc)
        {
            SourceNode sn = new SourceNode
            {
                ASTElement = isc,
                StartString = SwitchCaseStart(isc),
                EndString = SwitchCaseEnd(isc)
            };

            AddChild(sn);
            nodeStack.Push(sn);
            currTab++;
            AttachStatements(isc.Body.Statements);
            currTab--;
            nodeStack.Pop();
            return sn;
        }


        /// <summary>
        /// Attach catch clause
        /// </summary>
        /// <param name="icc">Catch clause</param>
        /// <returns>source node</returns>
        public virtual SourceNode AttachCatchClause(ICatchClause icc)
        {
            SourceNode sn = new SourceNode
            {
                ASTElement = icc,
                StartString = CatchClauseStart(icc),
                EndString = CatchClauseEnd(icc)
            };

            AddChild(sn);
            nodeStack.Push(sn);
            currTab++;
            AttachStatements(icc.Body.Statements);
            currTab--;
            nodeStack.Pop();
            return sn;
        }

        /// <summary>
        /// Attach a throw exception statement
        /// </summary>
        /// <param name="ites">Throw exception statement</param>
        /// <returns></returns>
        public virtual SourceNode AttachThrowExceptionStatement(IThrowExceptionStatement ites)
        {
            SourceNode sn = new SourceNode
            {
                ASTElement = ites,
                StartString = ThrowExceptionStart(ites),
                EndString = ThrowExceptionEnd(ites)
            };
            AddChild(sn);
            return sn;
        }

        /// <summary>
        /// Attach a Try/Catch/Finally statement
        /// </summary>
        /// <param name="itcfs">Try/Catch/Finally statement</param>
        /// <returns></returns>
        public virtual SourceNode AttachTryCatchFinallyStatement(ITryCatchFinallyStatement itcfs)
        {
            SourceNode sn = new SourceNode();
            sn.ASTElement = itcfs;
            AddChild(sn);
            nodeStack.Push(sn);

            // The try part of the statement
            SourceNode snTry = new SourceNode(TryBlockStart(itcfs), TryBlockEnd(itcfs), itcfs);
            AddChild(snTry);
            nodeStack.Push(snTry);
            currTab++;
            AttachStatements(itcfs.Try.Statements);
            currTab--;
            nodeStack.Pop();

            // The default catch part
            if (itcfs.Fault.Statements.Count > 0)
            {
                SourceNode snFault = new SourceNode(FaultBlockStart(itcfs), FaultBlockEnd(itcfs), itcfs);
                AddChild(snFault);
                nodeStack.Push(snFault);
                currTab++;
                AttachStatements(itcfs.Fault.Statements);
                currTab--;
                nodeStack.Pop();
            }

            // Any other catch clauses
            foreach (ICatchClause icc in itcfs.CatchClauses)
                AttachCatchClause(icc);

            // The finally part
            if (itcfs.Finally.Statements.Count > 0)
            {
                SourceNode snFinally = new SourceNode(FinallyBlockStart(itcfs), FinallyBlockEnd(itcfs), itcfs);
                AddChild(snFinally);
                nodeStack.Push(snFinally);
                currTab++;
                AttachStatements(itcfs.Finally.Statements);
                currTab--;
                nodeStack.Pop();
            }
            nodeStack.Pop();
            return sn;
        }

        /// <summary>
        /// Attach using statement
        /// </summary>
        /// <param name="ius">Using statement</param>
        /// <returns>source node</returns>
        public virtual SourceNode AttachUsingStatement(IUsingStatement ius)
        {
            SourceNode sn = new SourceNode
            {
                ASTElement = ius,
                StartString = UsingStatementStart(ius),
                EndString = UsingStatementEnd(ius)
            };
            AddChild(sn);
            nodeStack.Push(sn);
            currTab++;
            AttachStatements(ius.Body.Statements);
            currTab--;
            nodeStack.Pop();
            return sn;
        }

        /// <summary>
        /// Attach while statement
        /// </summary>
        /// <param name="iws">While statement</param>
        /// <returns>source node</returns>
        public virtual SourceNode AttachWhileStatement(IWhileStatement iws)
        {
            SourceNode sn = new SourceNode
            {
                StartString = WhileStatementStart(iws),
                EndString = WhileStatementEnd(iws),
                ASTElement = iws
            };
            AddChild(sn);
            nodeStack.Push(sn);
            currTab++;
            AttachStatements(iws.Body.Statements);
            currTab--;
            nodeStack.Pop();
            return sn;
        }

        /// <summary>
        /// Attach statement to the source tree
        /// </summary>
        /// <param name="ist">The statement</param>
        public virtual SourceNode AttachStatement(IStatement ist)
        {
            if (ist is IBlockStatement) return AttachBlockStatement(ist as IBlockStatement);
            else if (ist is IBreakStatement) return AttachBreakStatement(ist as IBreakStatement);
            else if (ist is ICommentStatement) return AttachCommentStatement(ist as ICommentStatement);
            else if (ist is IConditionStatement) return AttachConditionStatement(ist as IConditionStatement);
            else if (ist is IContinueStatement) return AttachContinueStatement(ist as IContinueStatement);
            else if (ist is IExpressionStatement) return AttachExpressionStatement(ist as IExpressionStatement);
            else if (ist is IForEachStatement) return AttachForEachStatement(ist as IForEachStatement);
            else if (ist is IForStatement) return AttachForStatement(ist as IForStatement);
            else if (ist is IRepeatStatement) return AttachRepeatStatement(ist as IRepeatStatement);
            else if (ist is IMethodReturnStatement) return AttachMethodReturnStatement(ist as IMethodReturnStatement);
            else if (ist is ISwitchStatement) return AttachSwitchStatement(ist as ISwitchStatement);
            else if (ist is IThrowExceptionStatement) return AttachThrowExceptionStatement(ist as IThrowExceptionStatement);
            else if (ist is ITryCatchFinallyStatement) return AttachTryCatchFinallyStatement(ist as ITryCatchFinallyStatement);
            else if (ist is IUsingStatement) return AttachUsingStatement(ist as IUsingStatement);
            else if (ist is IWhileStatement) return AttachWhileStatement(ist as IWhileStatement);
            else throw new NotSupportedException("Language writer: unsupported statement type");
        }

        /// <summary>
        /// Attach all statements to the source tree
        /// </summary>
        /// <param name="isc"></param>
        public void AttachStatements(IEnumerable<IStatement> isc)
        {
            int count = nodeStack.Count;
            foreach (IStatement ist in isc)
            {
                AttachStatement(ist);
                if (nodeStack.Count != count) throw new Exception("nodeStack didn't get popped");
            }
        }

        /// <summary>
        /// Source for a statement
        /// </summary>
        /// <param name="ist">Statement</param>
        public string StatementSource(IStatement ist)
        {
            Initialise();
            SourceNode root = new SourceNode();
            nodeStack.Push(root);
            return this.AttachStatement(ist).ToString(this);
        }

        /// <summary>
        /// Attach a method to the source tree
        /// </summary>
        /// <param name="imd">Method declaration</param>
        public virtual SourceNode AttachMethodDeclaration(IMethodDeclaration imd)
        {
            if (imd == null)
                return null;

            // Create the node...
            SourceNode sn = new SourceNode(MethodDeclarationStart(imd), MethodDeclarationEnd(imd), imd);
            AddChild(sn);
            nodeStack.Push(sn);
            currTab++;

            // Add the body
            IBlockStatement body = imd.Body;
            if (body != null)
                AttachStatements(body.Statements);

            currTab--;
            nodeStack.Pop();

            return sn;
        }

        /// <summary>
        /// Method declaration source
        /// </summary>
        /// <param name="imd">Method declaration</param>
        public string MethodDeclarationSource(IMethodDeclaration imd)
        {
            Initialise();
            SourceNode root = new SourceNode();
            nodeStack.Push(root);
            return AttachMethodDeclaration(imd).ToString(this);
        }

        /// <summary>
        /// Attach a field to the source tree
        /// </summary>
        /// <param name="ifd">Field declaration</param>
        public virtual SourceNode AttachFieldDeclaration(IFieldDeclaration ifd)
        {
            // The data for the tree node
            SourceNode sn = new SourceNode(FieldDeclarationStart(ifd), FieldDeclarationEnd(ifd), ifd);
            AddChild(sn);
            return sn;
        }

        /// <summary>
        /// Convert an individual field to source
        /// </summary>
        /// <param name="ifd">Field declaration</param>
        public string FieldDeclarationSource(IFieldDeclaration ifd)
        {
            Initialise();
            SourceNode root = new SourceNode();
            nodeStack.Push(root);
            return AttachFieldDeclaration(ifd).ToString(this);
        }

        /// <summary>
        /// Convert property collection to a source node and attach to
        /// cuurent node
        /// </summary>
        /// <param name="ipdc">Property collection</param>
        public virtual SourceNode AttachPropertyDeclarationCollection(List<IPropertyDeclaration> ipdc)
        {
            SourceNode sn = new SourceNode();
            sn.ASTElement = ipdc;
            if (ipdc != null && ipdc.Count > 0)
            {
                sn.StartString = PropertyDeclarationCollectionStart(ipdc);
                sn.EndString = PropertyDeclarationCollectionEnd(ipdc);
                sn.ASTElement = ipdc;
                AddChild(sn);
                nodeStack.Push(sn);
                foreach (IPropertyDeclaration ipd in ipdc)
                {
                    AttachPropertyDeclaration(ipd);
                }
                nodeStack.Pop();
            }
            return sn;
        }

        /// <summary>
        /// Convert properties to source
        /// </summary>
        /// <param name="ipdc">Property collection</param>
        public string PropertyDeclarationCollectionSource(List<IPropertyDeclaration> ipdc)
        {
            this.Initialise();
            SourceNode root = new SourceNode();
            nodeStack.Push(root);
            return this.AttachPropertyDeclarationCollection(ipdc).ToString(this);
        }

        /// <summary>
        /// Attach a property to the source tree
        /// </summary>
        /// <param name="ipd">Property declaration</param>
        public virtual SourceNode AttachPropertyDeclaration(IPropertyDeclaration ipd)
        {
            // Create the node
            SourceNode sn = new SourceNode(PropertyDeclarationStart(ipd), PropertyDeclarationEnd(ipd), ipd);
            AddChild(sn);
            nodeStack.Push(sn);
            currTab++;
            AttachPropertyAccessor(ipd.GetMethod, false);
            AttachPropertyAccessor(ipd.SetMethod, true);
            currTab--;
            nodeStack.Pop();

            return sn;
        }

        /// <summary>
        /// Generate source for a single property
        /// </summary>
        /// <param name="ipd">Property declaration</param>
        public string PropertyDeclarationSource(IPropertyDeclaration ipd)
        {
            this.Initialise();
            SourceNode root = new SourceNode();
            nodeStack.Push(root);
            return this.AttachPropertyDeclaration(ipd).ToString(this);
        }

        /// <summary>
        /// Convert event collection to a source node and attach to
        /// cuurent node
        /// </summary>
        /// <param name="iedc">Events collection</param>
        public virtual SourceNode AttachEventDeclarationCollection(List<IEventDeclaration> iedc)
        {
            SourceNode sn = new SourceNode();
            sn.ASTElement = iedc;
            if (iedc != null && iedc.Count > 0)
            {
                sn.StartString = EventDeclarationCollectionStart(iedc);
                sn.EndString = EventDeclarationCollectionEnd(iedc);
                sn.ASTElement = iedc;
                AddChild(sn);
                nodeStack.Push(sn);
                foreach (IEventDeclaration ied in iedc)
                {
                    AttachEventDeclaration(ied);
                }
                nodeStack.Pop();
            }
            return sn;
        }

        /// <summary>
        /// Convert events to source
        /// </summary>
        /// <param name="iedc">Event collection</param>
        public string EventDeclarationCollectionSource(List<IEventDeclaration> iedc)
        {
            this.Initialise();
            SourceNode root = new SourceNode();
            nodeStack.Push(root);
            return this.AttachEventDeclarationCollection(iedc).ToString(this);
        }

        /// <summary>
        /// Attach an event declaration to the source tree
        /// </summary>
        /// <param name="ied">Event declaration</param>
        public virtual SourceNode AttachEventDeclaration(IEventDeclaration ied)
        {
            // The data for the tree node
            SourceNode sn = new SourceNode(EventDeclarationStart(ied), EventDeclarationEnd(ied), ied);
            AddChild(sn);
            return sn;
        }

        /// <summary>
        /// Generate source for a single event
        /// </summary>
        /// <param name="ied">Event declaration</param>
        public string EventDeclarationSource(IEventDeclaration ied)
        {
            this.Initialise();
            SourceNode root = new SourceNode();
            nodeStack.Push(root);
            return this.AttachEventDeclaration(ied).ToString(this);
        }

        /// <summary>
        /// Attach a type collection to the source tree
        /// </summary>
        /// <param name="intdc">Type collection</param>
        public virtual SourceNode AttachTypeDeclarationCollection(List<ITypeDeclaration> intdc)
        {
            SourceNode sn = new SourceNode();
            sn.ASTElement = intdc;
            if (intdc != null && intdc.Count > 0)
            {
                // Create the node
                sn.StartString = TypeDeclarationCollectionStart(intdc);
                sn.EndString = TypeDeclarationCollectionEnd(intdc);
                sn.ASTElement = intdc;
                AddChild(sn);
                nodeStack.Push(sn);
                foreach (ITypeDeclaration intd in intdc)
                {
                    AttachTypeDeclaration(intd);
                }
                nodeStack.Pop();
            }
            return sn;
        }

        /// <summary>
        /// Convert type collection to source
        /// </summary>
        /// <param name="intdc">Type collection</param>
        public string TypeDeclarationCollectionSource(List<ITypeDeclaration> intdc)
        {
            this.Initialise();
            SourceNode root = new SourceNode();
            nodeStack.Push(root);
            return this.AttachTypeDeclarationCollection(intdc).ToString(this);
        }

        /// <summary>
        /// Attach field collection to source tree
        /// </summary>
        /// <param name="ifdc">Field collection</param>
        public virtual SourceNode AttachFieldDeclarationCollection(List<IFieldDeclaration> ifdc)
        {
            SourceNode sn = new SourceNode();
            sn.ASTElement = ifdc;
            if (ifdc != null && ifdc.Count > 0)
            {
                sn.StartString = FieldDeclarationCollectionStart(ifdc);
                sn.EndString = FieldDeclarationCollectionEnd(ifdc);
                sn.ASTElement = ifdc;
                AddChild(sn);
                nodeStack.Push(sn);
                foreach (IFieldDeclaration ifd in ifdc)
                {
                    AttachFieldDeclaration(ifd);
                }
                nodeStack.Pop();
            }
            return sn;
        }

        /// <summary>
        /// Convert fields to source
        /// </summary>
        /// <param name="ifdc">Field collection</param>
        public string FieldDeclarationCollectionSource(List<IFieldDeclaration> ifdc)
        {
            this.Initialise();
            SourceNode root = new SourceNode();
            nodeStack.Push(root);
            return this.AttachFieldDeclarationCollection(ifdc).ToString(this);
        }

        /// <summary>
        /// Attach method collection to the source tree
        /// </summary>
        /// <param name="imdc">Method collection</param>
        public virtual SourceNode AttachMethodDeclarationCollection(List<IMethodDeclaration> imdc)
        {
            SourceNode sn = new SourceNode();
            sn.ASTElement = imdc;
            if (imdc != null && imdc.Count > 0)
            {
                // Create the node
                sn.StartString = MethodDeclarationCollectionStart(imdc);
                sn.EndString = MethodDeclarationCollectionEnd(imdc);
                sn.ASTElement = imdc;
                AddChild(sn);
                nodeStack.Push(sn);
                foreach (IMethodDeclaration imd in imdc)
                {
                    AttachMethodDeclaration(imd);
                }
                nodeStack.Pop();
            }
            return sn;
        }

        /// <summary>
        /// Convert methods to source
        /// </summary>
        /// <param name="imdc">Method collection</param>
        public string MethodDeclarationCollectionSource(List<IMethodDeclaration> imdc)
        {
            this.Initialise();
            SourceNode root = new SourceNode();
            nodeStack.Push(root);
            return this.AttachMethodDeclarationCollection(imdc).ToString(this);
        }

        /// <summary>
        /// Write a source node to a string
        /// </summary>
        /// <param name="sw">String writer</param>
        /// <param name="sn">Source node</param>
        public static void WriteSourceNode(TextWriter sw, SourceNode sn)
        {
            sw.Write(sn.StartString);
            if (sn.Children != null)
            {
                foreach (SourceNode child in sn.Children)
                    WriteSourceNode(sw, child);
            }
            sw.Write(sn.EndString);
        }
    }
}