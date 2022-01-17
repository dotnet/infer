// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Compiler
{
    using System;
    using System.Collections.Generic;
    using System.Text;
    using System.Reflection;
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Utilities;
    using Microsoft.ML.Probabilistic.Compiler.CodeModel;
    using Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete;
    using System.Globalization;
    using System.Collections.Concurrent;

/// <summary>
/// Singleton class to help build class declarations.
/// </summary>
    public partial class CodeBuilder
    {
        /// <summary>
        /// Constructor
        /// </summary>
        protected CodeBuilder()
        {
        }

        /// <summary>
        /// Converts a code model type into a System.Reflection Type.
        /// </summary>
        public Type ToType(IDotNetType t)
        {
            return t.DotNetType;
        }

        /// <summary>
        /// Get assembly qualified dotNET type name
        /// </summary>
        /// <param name="it">Code mode type</param>
        /// <returns></returns>
        protected string ToTypeName(IDotNetType it)
        {
            Type tp = it.DotNetType;

            if (tp == null)
                return "";
            else
                return tp.AssemblyQualifiedName;
        }

        /// <summary>
        /// Converts a code model method reference to a MethodBase
        /// </summary>
        /// <param name="imr">Method reference</param>
        /// <returns>Returns MethodBase if conversion is successful, null otherwise</returns>
        public MethodBase ToMethod(IMethodReference imr)
        {
            return imr.MethodInfo;
            //return ToMethodThrows(imr);
        }

        /// <summary>
        /// Converts a code model method reference to a MethodBase
        /// </summary>
        /// <param name="imr">Method reference</param>
        /// <returns>Returns MethodBase if conversion is successful, throws an exception otherwise</returns>
        public MethodBase ToMethodThrows(IMethodReference imr)
        {
            if (imr.MethodInfo == null) imr.MethodInfo = ToMethodInternal(imr);
            return imr.MethodInfo;
        }

        /// <summary>
        /// Convert a type reference to a MethodInfo
        /// </summary>
        /// <param name="imr"></param>
        /// <returns></returns>
        internal MethodBase ToMethodInternal(IMethodReference imr)
        {
            if (imr == null)
                return null;


            IMethodDeclaration imd = imr.Resolve();

            Type[] parameterTypes = new Type[imr.Parameters.Count];
            Type returnType = ToType(imr.ReturnType.Type);

            for (int i = 0; i < parameterTypes.Length; i++)
            {
                IType ipt = imr.Parameters[i].ParameterType;
                parameterTypes[i] = ToType(ipt);
                if (parameterTypes[i] == null)
                    // This is a generic method. We won't deal with this here.
                    // Rather, when we find an instance, will hook up the method
                    // info at that point
                    return null;
            }
            Type tp = ToType(imr.DeclaringType);
            if (tp == null)
                return null;
            // TODO: handle .cctor case (static initialiser??)
            if (imr.Name.Equals(".ctor") || imr.Name.Equals(".cctor"))
            {
                ConstructorInfo ci = null;
                try
                {
                    ci = tp.GetConstructor(parameterTypes);
                }
                catch
                {
                }
                return ci;
            }

            MethodInfo mi = null;
            BindingFlags bf = BindingFlags.FlattenHierarchy | BindingFlags.IgnoreReturn;
            int nPass = 2;
            if (imd != null)
            {
                if (imd.Visibility == MethodVisibility.Private)
                    bf |= BindingFlags.NonPublic;
                else
                    bf |= BindingFlags.Public;
                if (imd.Static)
                    bf |= BindingFlags.Static;
                else
                    bf |= BindingFlags.Instance;
            }
            else
            {
                bf |=
                    BindingFlags.Public |
                    BindingFlags.NonPublic |
                    BindingFlags.Static |
                    BindingFlags.Instance;
                nPass = 1;
            }

            try
            {
                if (imr.GenericArguments.Count <= 0)
                {
                    for (int pass = 0; pass < nPass; pass++)
                    {
                        if (pass == 1)
                        {
                            // cast a wider net on the second pass
                            bf |= (BindingFlags.NonPublic | BindingFlags.Public);
                        }
                        try
                        {
                            // If this is not a generic instance, we should be able to directly get the type
                            // Skip the ones that we know are going to fail
                            if (!imr.Name.Equals("op_Explicit") &&
                                !imr.Name.Equals("Marginal") &&
                                !imr.Name.Equals("GetKey") &&
                                //!imr.Name.Equals("Infer") &&
                                !imr.Name.Equals("GetOpenBlocks") &&
                                !imr.Name.Equals("ToArray"))
                            {
                                mi = tp.GetMethod(imr.Name, bf, null, parameterTypes, null);
                                if ((mi == null) && (imr.Parameters.Count == 0))
                                {
                                    // we may have a method reference which is just by name
                                    mi = tp.GetMethod(imr.Name);
                                }
                                if (mi != null)
                                    break;
                            }
                        }
                        catch
                        {
                        }
                        // Do it the long way
                        mi = FindNonGenericMethod(tp, imr.Name, parameterTypes, returnType, bf);
                        if (mi != null)
                            break;
                    }
                    return mi;
                }
                else
                {
                    Type[] typeArgs = new Type[imr.GenericArguments.Count];
                    for (int i = 0; i < typeArgs.Length; i++)
                    {
                        typeArgs[i] = ToType(imr.GenericArguments[i]);
                        if (typeArgs[i] == null)
                            return null;
                    }

                    bf |= BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Public | BindingFlags.Instance;
                    mi = FindGenericMethod(tp, imr.Name, typeArgs, parameterTypes, returnType, bf);
                }
            }
            catch (Exception)
            {
            }
            return mi;
        }

        /// <summary>
        /// Find non-generic method
        /// </summary>
        /// <param name="tp">Declaring type</param>
        /// <param name="methodName">Method name</param>
        /// <param name="parameterTypes">Parameter types</param>
        /// <param name="returnType">Return type - needed for some internal signatures</param>
        /// <param name="bf">Binding flags</param>
        /// <returns></returns>
        private static MethodInfo FindNonGenericMethod(
            Type tp,
            string methodName,
            Type[] parameterTypes,
            Type returnType,
            BindingFlags bf)
        {
            MethodInfo[] methods = tp.GetMethods(bf);
            foreach (MethodInfo method in methods)
            {
                if (method.Name != methodName)
                    continue;

                ParameterInfo[] parameters = method.GetParameters();
                if (parameters.Length != parameterTypes.Length)
                    continue;

                for (int i = 0; i < parameters.Length; i++)
                {
                    if (parameters[i].ParameterType != parameterTypes[i])
                        continue;
                }
                if (returnType != null && method.ReturnType != null)
                {
                    // In some weird cases, this is needed - op_Explicit for example
                    if (method.ReturnType.Name != returnType.Name)
                        continue;
                }

                return method;
            }
            return null;
        }

        /// <summary>
        /// Find a method from its type method name, type arguments and parameter types
        /// </summary>
        /// <param name="type">The declaring type of the method</param>
        /// <param name="methodName">The name of the method</param>
        /// <param name="typeArguments">The type arguments</param>
        /// <param name="parameterTypes">The parameter types</param>
        /// <param name="returnType">The return type</param>
        /// <param name="bf">The binding flags</param>
        /// <returns></returns>
        private static MethodInfo FindGenericMethod(
            Type type,
            string methodName,
            Type[] typeArguments,
            Type[] parameterTypes,
            Type returnType,
            BindingFlags bf)
        {
            MethodInfo[] methods = type.GetMethods(bf);
            foreach (MethodInfo method in methods)
            {
                if (method.Name != methodName)
                    continue;

                MethodInfo genericMethod = null;
                if (method.IsGenericMethod) //&& !method.IsGenericMethodDefinition)
                {
                    MethodInfo genericMethodDef = method.GetGenericMethodDefinition();
                    Type[] genParms = genericMethodDef.GetGenericArguments();
                    if (genParms.Length != typeArguments.Length)
                        continue;
                    genericMethod = method.MakeGenericMethod(typeArguments);
                }
                else
                {
                    genericMethod = method;
                }
                ParameterInfo[] parameters = genericMethod.GetParameters();

                // compare the method parameters
                if (parameters.Length != parameterTypes.Length) continue;

                for (int i = 0; i < parameters.Length; i++)
                {
                    if (parameters[i].ParameterType != parameterTypes[i])
                        continue;
                }
                if (returnType != null)
                {
                    if (!genericMethod.ReturnType.IsGenericParameter &&
                        !returnType.IsGenericParameter)
                    {
                        if (genericMethod.ReturnType.Name != returnType.Name)
                            continue;
                    }
                }
                // if we're here, we got the right method.
                return genericMethod;
            }
            return null;
        }


        /// <summary>
        /// Creates a new field declaration
        /// </summary>
        /// <param name="name">Name of the field</param>
        /// <param name="type">Code model type</param>
        /// <param name="declaringType">Declaring type</param>
        /// <returns>Field declaration</returns>
        public IFieldDeclaration FieldDecl(string name, Type type, IType declaringType)
        {
            return FieldDecl(name, TypeRef(type), declaringType);
        }

        /// <summary>
        /// Creates a new field declaration
        /// </summary>
        /// <param name="name">Name of the field</param>
        /// <param name="type">Code model type</param>
        /// <param name="declaringType">Declaring type</param>
        /// <returns>Field declaration</returns>
        public IFieldDeclaration FieldDecl(string name, IType type, IType declaringType)
        {
            return FieldDecl(name, type, declaringType, null);
        }

        /// <summary>
        /// Creates a new field declaration
        /// </summary>
        /// <param name="name">Name of the field</param>
        /// <param name="type">Code model type</param>
        /// <param name="declaringType">Declaring type</param>
        /// <param name="initializer">Initializer expression</param>
        /// <returns>Field declaration</returns>
        private IFieldDeclaration FieldDecl(string name, IType type, IType declaringType, IExpression initializer)
        {
            IFieldDeclaration fd = FieldDecl();
            fd.DeclaringType = declaringType;
            fd.Name = name;
            fd.Initializer = initializer;
            //modelBuilderField.Static = true;
            fd.Visibility = FieldVisibility.Private;
            fd.FieldType = type;
            return fd;
        }

        /// <summary>
        /// Creates a field reference expression
        /// </summary>
        /// <param name="fr">Field reference</param>
        /// <returns>Field reference expression</returns>
        public IFieldReferenceExpression FieldRefExpr(IFieldReference fr)
        {
            IFieldReferenceExpression fre = FieldRefExpr();
            fre.Field = fr;
            fre.Target = ThisRefExpr();
            return fre;
        }

        /// <summary>
        /// Creates a field reference expression
        /// </summary>
        /// <param name="target">Expression for type instance</param>
        /// <param name="declaringType">Declaring type</param>
        /// <param name="name">Field name</param>
        /// <returns>Field reference expression</returns>
        public IFieldReferenceExpression FieldRefExpr(IExpression target, Type declaringType, string name)
        {
            return FieldRefExpr(target, TypeRef(declaringType),
                                TypeRef(declaringType.GetField(name).FieldType), name);
        }

        /// <summary>
        /// Creates a field reference expression
        /// </summary>
        /// <param name="target">Expression for type instance</param>
        /// <param name="declaringType">Declaring type</param>
        /// <param name="fieldType">Type of field</param>
        /// <param name="name">Field name</param>
        /// <returns>Field reference expression</returns>
        public IFieldReferenceExpression FieldRefExpr(IExpression target, IType declaringType, IType fieldType, string name)
        {
            IFieldReferenceExpression fre = FieldRefExpr();
            IFieldReference fr = FieldRef();
            fr.Name = name;
            fr.FieldType = fieldType;
            fr.DeclaringType = declaringType;
            fre.Field = fr;
            fre.Target = target;
            return fre;
        }

        /// <summary>
        /// Creates an 'is' expression
        /// </summary>
        /// <param name="expr">Expression</param>
        /// <param name="t">Type to test against</param>
        /// <returns>'is' expression</returns>
        public ICanCastExpression CanCastExpr(IExpression expr, Type t)
        {
            return CanCastExpr(expr, TypeRef(t));
        }

        /// <summary>
        /// Creates an 'is' expression
        /// </summary>
        /// <param name="expr">Expression</param>
        /// <param name="t">Type to test against</param>
        /// <returns>'is' expression</returns>
        public ICanCastExpression CanCastExpr(IExpression expr, IType t)
        {
            ICanCastExpression ce = new XCanCastExpression();
            ce.Expression = expr;
            ce.TargetType = t;
            return ce;
        }

        /// <summary>
        /// Creates a cast expression
        /// </summary>
        /// <param name="expr">Expression</param>
        /// <param name="t">Type to cast it to</param>
        /// <returns>Cast expression</returns>
        public ICastExpression CastExpr(IExpression expr, Type t)
        {
            return CastExpr(expr, TypeRef(t));
        }

        /// <summary>
        /// Creates a cast expression
        /// </summary>
        /// <param name="expr">Expression</param>
        /// <param name="t">Type to cast it to</param>
        /// <returns>Cast expression</returns>
        public ICastExpression CastExpr(IExpression expr, IType t)
        {
            ICastExpression ce = CastExpr();
            ce.Expression = expr;
            ce.TargetType = t;
            return ce;
        }

        /// <summary>
        /// Creates a checked arithmetic expression
        /// </summary>
        /// <param name="expr">Expression</param>
        /// <returns>Checked expression</returns>
        public ICheckedExpression CheckedExpr(IExpression expr)
        {
            ICheckedExpression ce = CheckedExpr();
            ce.Expression = expr;
            return ce;
        }

        /// <summary>
        /// Creates an expression statement
        /// </summary>
        /// <param name="expr">The expression</param>
        /// <returns>Expression statement</returns>
        public IExpressionStatement ExprStatement(IExpression expr)
        {
            if (expr == null) throw new ArgumentNullException(nameof(expr));
            IExpressionStatement es = ExprStatement();
            es.Expression = expr;
            return es;
        }

        private static readonly ConcurrentDictionary<Type, IType> typeCache = new ConcurrentDictionary<Type, IType>();

        /// <summary>
        /// Creates a type reference
        /// </summary>
        /// <param name="t">The dotNET type</param>
        /// <returns></returns>
        public IType TypeRef(Type t)
        {
            IType it;
            if (typeCache.TryGetValue(t, out it)) return it;
            if (t.IsArray)
            {
                it = ArrayType(TypeRef(t.GetElementType()), t.GetArrayRank());
            }
            else if (t.IsGenericType && (!t.ContainsGenericParameters))
            {
                Type[] args = t.GetGenericArguments();
                IType[] genArgs = Array.ConvertAll(args, arg => TypeRef(arg));
                it = TypeRef(t.Name, t, null, genArgs);
            }
            else
            {
                // TODO: make this work correctly for nested classes
                it = TypeRef(t.Name, t, null);
            }
            typeCache[t] = it;
            return it;
        }

        /// <summary>
        /// Creates a type reference
        /// </summary>
        /// <param name="name">Name of the reference</param>
        /// <param name="t">Type</param>
        /// <param name="outerClass">The owner type if this is a generic parameter type, otherwise null</param>
        /// <param name="typeArguments">Type arguments</param>
        /// <returns>Type reference</returns>
        public ITypeReference TypeRef(string name, Type t, IType outerClass, params IType[] typeArguments)
        {
            ITypeReference tir = TypeInstRef();
            foreach (IType tp in typeArguments) tir.GenericArguments.Add(tp);
            string s = "`" + typeArguments.Length;
            if (name.EndsWith(s)) name = name.Substring(0, name.Length - s.Length);
            tir.GenericType = (ITypeReference)TypeRef(name, t, outerClass);
            if (tir is IDotNetType tir_dn)
                tir_dn.DotNetType = t;
            return tir;
        }

        /// <summary>
        /// Creates a type reference
        /// </summary>
        /// <param name="name">Name of the reference</param>
        /// <param name="t">Type</param>
        /// <param name="outerClass">The owner type if this is a generic parameter type, otherwise null</param>
        /// <returns>Type reference</returns>
        public IType TypeRef(string name, Type t, IType outerClass)
        {
            ITypeReference tr = TypeRef();
            name = name.Replace('+', '.');
            bool isByRef = (name.EndsWith("&"));
            if (isByRef) name = name.Substring(0, name.Length - 1);
            //tr.Name = name;
            //tr.Namespace = t.Namespace;
            if (t != null && t.Assembly != null)
            {
                // TM: Owner = System.Reflection.Assembly seems to work just as well as IModuleReference.
                tr.Owner = t.Assembly;
                //tr.Owner = AssemblyRef(module, assembly);
            }
            if (outerClass != null)
                tr.Owner = outerClass;

            if (tr is IDotNetType tr_dn)
                tr_dn.DotNetType = t;

            if (isByRef)
            {
                IReferenceType rt = RefType();
                rt.ElementType = tr;
                if (rt is IDotNetType rt_dn)
                    rt_dn.DotNetType = t;
                return rt;
            }
            return tr;
        }

        /// <summary>
        /// Create a type reference expression
        /// </summary>
        /// <param name="type">A type reference</param>
        /// <returns>Type reference expression</returns>
        public ITypeReferenceExpression TypeRefExpr(IType type)
        {
            ITypeReferenceExpression itre = TypeRefExpr();
            itre.Type = (ITypeReference)type;
            return itre;
        }

        /// <summary>
        /// Create a type reference expression
        /// </summary>
        /// <param name="type">A type</param>
        /// <returns>Type reference expression</returns>
        public ITypeReferenceExpression TypeRefExpr(Type type)
        {
            return TypeRefExpr(TypeRef(type));
        }

        /// <summary>
        /// Creates a call to a static method
        /// </summary>
        /// <param name="imr">Method reference</param>
        /// <param name="args">Argument expressions</param>
        /// <returns>The method invoke expression</returns>
        public IMethodInvokeExpression StaticMethod(IMethodReference imr, params IExpression[] args)
        {
            return Method(TypeRefExpr(imr.DeclaringType), imr, args);
        }

        /// <summary>
        /// Creates a call to a static method
        /// </summary>
        /// <param name="d">Delegate for the method</param>
        /// <param name="args">Argument expressions</param>
        /// <returns>The method invoke expression</returns>
        public IMethodInvokeExpression StaticMethod(Delegate d, params IExpression[] args)
        {
            return StaticMethod(d.Method, args);
        }

        /// <summary>
        /// Creates a call to a static method
        /// </summary>
        /// <param name="mi">MethodInfo instance</param>
        /// <param name="args">Argument expressions</param>
        /// <returns>The method invoke expression</returns>
        public IMethodInvokeExpression StaticMethod(MethodInfo mi, params IExpression[] args)
        {
            return Method(TypeRefExpr(mi.DeclaringType), mi, args);
        }

        /// <summary>
        /// Creates a call to a method
        /// </summary>
        /// <param name="target">Instance for method call</param>
        /// <param name="d">Delegate for the method</param>
        /// <param name="args">Argument expressions</param>
        /// <returns>The method invoke expression</returns>
        public IMethodInvokeExpression Method(IExpression target, Delegate d, params IExpression[] args)
        {
            return Method(target, d.Method, args);
        }

        /// <summary>
        /// Creates a call to a method
        /// </summary>
        /// <param name="target">Instance for method call</param>
        /// <param name="mi">Method info</param>
        /// <param name="args">Argument expressions</param>
        /// <returns>The method invoke expression</returns>
        public IMethodInvokeExpression Method(IExpression target, MethodInfo mi, params IExpression[] args)
        {
            IMethodInvokeExpression mie = MethodInvkExpr();
            mie.Method = MethodRefExpr(GenericMethodRef(mi), target);
            ParameterInfo[] parameters = mi.GetParameters();
            bool hasParamsArgument = false;
            for (int i = 0; i < args.Length; i++)
            {
                IExpression expr = args[i];
                if (i < parameters.Length)
                {
                    var parameter = parameters[i];
                    if (parameter.IsOut)
                    {
                        IAddressOutExpression aoe = AddrOutExpr();
                        aoe.Expression = expr;
                        expr = aoe;
                    }
                    else if (parameter.ParameterType.IsByRef)
                    {
                        IAddressReferenceExpression are = AddrRefExpr();
                        are.Expression = expr;
                        expr = are;
                    }
                }
                else if (!hasParamsArgument)
                {
                    // check that the method has a final 'params' argument
                    var parameter = parameters[parameters.Length - 1];
                    hasParamsArgument = parameter.GetCustomAttributes(typeof(ParamArrayAttribute), true).Length > 0;
                    if (!hasParamsArgument)
                    {
                        throw new ArgumentException("Too many arguments provided");
                    }
                }
                mie.Arguments.Add(expr);
            }
            return mie;
        }

        /// <summary>
        /// Creates a call to a method
        /// </summary>
        /// <param name="target">Instance for method call</param>
        /// <param name="imd">Method declaration</param>
        /// <param name="args">Argument expressions</param>
        /// <returns>The method invoke expression</returns>
        public IMethodInvokeExpression Method(IExpression target, IMethodDeclaration imd, params IExpression[] args)
        {
            IMethodInvokeExpression mie = MethodInvkExpr();
            mie.Method = MethodRefExpr(imd, target);
            mie.Arguments.AddRange(args);
            return mie;
        }

        /// <summary>
        /// Creates a call to a method
        /// </summary>
        /// <param name="target">Instance for method call</param>
        /// <param name="imr">Method reference</param>
        /// <param name="args">Argument expressions</param>
        /// <returns>The method invoke expression</returns>
        public IMethodInvokeExpression Method(IExpression target, IMethodReference imr, params IExpression[] args)
        {
            IMethodInvokeExpression mie = MethodInvkExpr();
            mie.Method = MethodRefExpr(imr, target);
            mie.Arguments.AddRange(args);
            return mie;
        }

        /// <summary>
        /// Creates a call to a static generic method
        /// </summary>
        /// <param name="d">The delegate</param>
        /// <param name="args">Argument expressions</param>
        /// <returns>The method invoke expression</returns>
        public IMethodInvokeExpression StaticGenericMethod(Delegate d, params IExpression[] args)
        {
            return StaticGenericMethod(d.Method, args);
        }

        /// <summary>
        /// Creates a call to a static generic method
        /// </summary>
        /// <param name="mi">The MethodInfo</param>
        /// <param name="args">Argument expressions</param>
        /// <returns>The method invoke expression</returns>
        public IMethodInvokeExpression StaticGenericMethod(MethodInfo mi, params IExpression[] args)
        {
            return StaticMethod(mi, args);
        }

        /// <summary>
        /// Creates a call to a static generic method by replacing the type arguments in the method referred to by the delegate
        /// with the specified arguments.  This allows the method to be referred to statically even when the generic arguments
        /// are not known at compile-time.  By convention, the type arguments of the delegate method should be object or the highest
        /// level class which satisifies the type constraints.
        /// </summary>
        /// <param name="d">Delegate</param>
        /// <param name="replacementGenericArgs">Replacement generic arguments</param>
        /// <param name="args">Argument expressions</param>
        /// <returns>The method invoke expression</returns>
        /// <exception cref="InvalidOperationException"></exception>
        /// <exception cref="ArgumentNullException"></exception>
        /// <exception cref="ArgumentException"></exception>
        public IMethodInvokeExpression StaticGenericMethod(Delegate d, Type[] replacementGenericArgs, params IExpression[] args)
        {
            MethodInfo mi = d.Method.GetGenericMethodDefinition().MakeGenericMethod(replacementGenericArgs);
            return StaticMethod(mi, args);
        }

        /// <summary>
        /// Creates a call to a static generic method by replacing the type arguments in the method referred to by the delegate
        /// with the specified arguments.  This allows the method to be referred to statically even when the generic arguments
        /// are not known at compile-time.  By convention, the type arguments of the delegate method should be object or the highest
        /// level class which satisifies the type constraints.
        /// </summary>
        /// <param name="d">Delegate</param>
        /// <param name="replacementGenericArgs">Replacement generic arguments</param>
        /// <param name="args">Argument expressions</param>
        /// <returns>The method invoke expression</returns>
        public IMethodInvokeExpression StaticGenericMethod(Delegate d, IType[] replacementGenericArgs, params IExpression[] args)
        {
            MethodInfo mi = d.Method;
            IMethodReference imr = GenericMethodRef(mi);
            imr.GenericArguments.Clear();
            foreach (IType arg in replacementGenericArgs)
            {
                imr.GenericArguments.Add(arg);
            }
            return StaticMethod(imr, args);
        }

        /// <summary>
        /// Creates an array of type references
        /// </summary>
        /// <param name="types">List of types</param>
        /// <returns>The array of types</returns>
        public IType[] TypeRefArray(IList<Type> types)
        {
            IType[] itypes = new IType[types.Count];
            for (int i = 0; i < types.Count; i++)
            {
                itypes[i] = TypeRef(types[i]);
            }
            return itypes;
        }

        /// <summary>
        /// Creates a method reference
        /// </summary>
        /// <param name="mi">MethodInfo</param>
        /// <returns>Method reference</returns>
        public IMethodReference MethodRef(MethodInfo mi)
        {
            IMethodReference m = MethodRef();
            m.Name = mi.Name;
            m.ReturnType = MethodReturnType(TypeRef(mi.ReturnType));
            ParameterInfo[] pis = mi.GetParameters();
            foreach (ParameterInfo pi in pis)
            {
                IParameterDeclaration pd = ParamDecl();
                pd.ParameterType = TypeRef(pi.ParameterType);
                pd.Name = pi.Name;
                m.Parameters.Add(pd);
            }
            m.DeclaringType = TypeRef(mi.DeclaringType);
            m.MethodInfo = mi;
            return m;
        }

        /// <summary>
        /// Creates a generic method reference
        /// </summary>
        /// <param name="mi">MethodInfo</param>
        /// <returns>Method reference</returns>
        public IMethodReference GenericMethodRef(MethodInfo mi)
        {
            // Get the parameter types
            ParameterInfo[] parmInfo = mi.GetParameters();
            int parmCnt = parmInfo != null ? parmInfo.Length : 0;
            // Create the Parameter Info array
            IType[] parmTypes = new IType[parmCnt];
            for (int i = 0; i < parmCnt; i++)
            {
                parmTypes[i] = TypeRef(parmInfo[i].ParameterType);
            }
            // Create the method reference
            IMethodReference m = MethodInstRef(parmTypes);
            m.GenericArguments.AddRange(TypeRefArray(mi.GetGenericArguments()));
            m.MethodInfo = mi;
            m.GenericMethod = MethodRef(mi);
            //m.GenericMethod.MethodInfo = mi.GetGenericMethodDefinition();
            m.DeclaringType = TypeRef(mi.DeclaringType);
            return m;
        }

        /// <summary>
        /// Creates a constructor declaration
        /// </summary>
        /// <param name="vis">Visibility of the constructor</param>
        /// <param name="declaringType">Type being constructed</param>
        /// <param name="pars">Parameters of the constructor</param>
        /// <returns>The constructor declaration</returns>
        public IConstructorDeclaration ConstructorDecl(MethodVisibility vis, IType declaringType, params IParameterDeclaration[] pars)
        {
            IConstructorDeclaration md = ConstructorDecl();
            md.DeclaringType = declaringType;
            md.Visibility = vis;
            //md.ReturnType = MethodRtrnTyp();
            //md.ReturnType.Type = declaringType;
            md.Body = BlockStmt();
            foreach (IParameterDeclaration ipd in pars) md.Parameters.Add(ipd);
            return md;
        }

        /// <summary>
        /// Creates a constructor reference
        /// </summary>
        /// <param name="t">Type</param>
        /// <param name="types">Parameter types for the constructor</param>
        /// <returns>The constructor reference</returns>
        public IMethodReference ConstructorRef(Type t, Type[] types)
        {
            ConstructorInfo mi;
            if (typeof(Delegate).IsAssignableFrom(t))
            {
                mi = t.GetConstructors()[0];
            }
            else
            {
                mi = t.GetConstructor(types);
            }
            if (mi == null && types.Length > 0)
                throw new MissingMethodException($"GetConstructor failed for type {StringUtil.TypeToString(t)} with arguments {StringUtil.ArrayToString(types)}");
            IMethodReference m = MethodRef();
            m.ReturnType = MethodReturnType(TypeRef(t));
            m.DeclaringType = TypeRef(t);
            m.MethodInfo = mi;
            return m;
        }

        /// <summary>
        /// Create a method declaration
        /// </summary>
        /// <param name="vis">Method visibility</param>
        /// <param name="name">Method name</param>
        /// <param name="returnType">Method's return type</param>
        /// <param name="declaringType">Method's declaring type</param>
        /// <param name="pars">Method parameters</param>
        /// <returns>The method declaration</returns>
        public IMethodDeclaration MethodDecl(MethodVisibility vis, string name, Type returnType, IType declaringType, params IParameterDeclaration[] pars)
        {
            return MethodDecl(vis, name, TypeRef(returnType), declaringType, pars);
        }

        /// <summary>
        /// Create a method declaration
        /// </summary>
        /// <param name="vis">Method visibility</param>
        /// <param name="name">Method name</param>
        /// <param name="returnType">Method's return code model type</param>
        /// <param name="declaringType">Method's declaring code model type</param>
        /// <param name="pars">Method parameters</param>
        /// <returns>The method declaration</returns>
        public IMethodDeclaration MethodDecl(MethodVisibility vis, string name, IType returnType, IType declaringType, params IParameterDeclaration[] pars)
        {
            if (returnType == null) returnType = TypeRef(typeof(void));
            IMethodDeclaration md = MethodDecl();
            md.DeclaringType = declaringType;
            md.Name = name;
            md.Visibility = vis;
            md.ReturnType = MethodReturnType(returnType);
            md.Body = BlockStmt();
            foreach (IParameterDeclaration ipd in pars) md.Parameters.Add(ipd);
            return md;
        }

        /// <summary>
        /// Create a generic method declaration
        /// </summary>
        /// <param name="vis">Method visibility</param>
        /// <param name="name">Method name</param>
        /// <param name="returnType">Method's return code model type</param>
        /// <param name="declaringType">Method's declaring code model type</param>
        /// <param name="genericParams">Type parameters in order</param>
        /// <param name="pars">Method parameters</param>
        /// <returns>The method declaration</returns>
        public IMethodDeclaration GenericMethodDecl(MethodVisibility vis, string name, IType returnType, IType declaringType, IEnumerable<IGenericParameter> genericParams,
                                                    params IParameterDeclaration[] pars)
        {
            if (returnType == null) returnType = TypeRef(typeof(void));
            IMethodDeclaration md = MethodDecl();
            md.DeclaringType = declaringType;
            md.Name = name;
            md.Visibility = vis;
            md.ReturnType = MethodReturnType(returnType);
            md.Body = BlockStmt();
            int gpx = 0;
            foreach (IGenericParameter igp in genericParams)
            {
                igp.Position = gpx++;
                igp.Owner = md;
                md.GenericArguments.Add(igp);
            }
            foreach (IParameterDeclaration ipd in pars) md.Parameters.Add(ipd);
            return md;
        }

        /// <summary>
        /// Creates a return statement
        /// </summary>
        /// <param name="expr">Expression for return statement</param>
        /// <returns>The method return statement</returns>
        public IMethodReturnStatement Return(IExpression expr)
        {
            IMethodReturnStatement mrs = MethodRtrnStmt();
            mrs.Expression = expr;
            return mrs;
        }

        /// <summary>
        /// Creates a return statement with no arguments
        /// </summary>
        /// <returns>The method return statement</returns>
        public IMethodReturnStatement Return()
        {
            IMethodReturnStatement mrs = MethodRtrnStmt();
            return mrs;
        }

        /// <summary>
        /// Creates a literal expression
        /// </summary>
        /// <param name="value">The value of the literal expression</param>
        /// <returns>The literal expression</returns>
        public ILiteralExpression LiteralExpr(object value)
        {
            ILiteralExpression le = LiteralExpr();
            le.Value = value;
            return le;
        }

        /// <summary>
        /// Creates a binary expression
        /// </summary>
        /// <param name="left">The expression to the left of the binary operator</param>
        /// <param name="op">The binary operator</param>
        /// <param name="right">The expression to the right of the binary operator</param>
        /// <returns>The binary expression</returns>
        public IBinaryExpression BinaryExpr(IExpression left, BinaryOperator op, IExpression right)
        {
            IBinaryExpression be = BinaryExpr();
            be.Left = left;
            be.Operator = op;
            be.Right = right;
            return be;
        }

        /// <summary>
        /// Creates a unary expression
        /// </summary>
        /// <param name="op">The unary operator</param>
        /// <param name="expr">The input expression</param>
        /// <returns></returns>
        public IUnaryExpression UnaryExpr(UnaryOperator op, IExpression expr)
        {
            IUnaryExpression ue = UnaryExpr();
            ue.Operator = op;
            ue.Expression = expr;
            return ue;
        }

        /// <summary>
        /// Creates an expression which is the sum of the given expressions
        /// </summary>
        /// <param name="exprs">Input expression</param>
        /// <returns></returns>
        public IExpression Add(params IExpression[] exprs)
        {
            return BinaryExpr(BinaryOperator.Add, exprs);
        }

        /// <summary>
        /// Creates an expression consisting of combining several expressions using
        /// a given binary operator
        /// </summary>
        /// <param name="op">The binary operator</param>
        /// <param name="exprs">The expressions</param>
        /// <returns></returns>
        public IExpression BinaryExpr(BinaryOperator op, params IExpression[] exprs)
        {
            IExpression result = exprs[0];
            for (int i = 1; i < exprs.Length; i++)
            {
                result = BinaryExpr(result, op, exprs[i]);
            }
            return result;
        }

        /// <summary>
        /// Creates a boolean NOT expression
        /// </summary>
        /// <param name="expr">The expression to the right of the operator</param>
        /// <returns>The unary expression</returns>
        public IUnaryExpression NotExpr(IExpression expr)
        {
            IUnaryExpression ue = UnaryExpr();
            ue.Operator = UnaryOperator.BooleanNot;
            ue.Expression = expr;
            return ue;
        }

        /// <summary>
        /// Creates an anonymous method expression
        /// </summary>
        /// <param name="delegateType">The delegate type</param>
        /// <returns></returns>
        public IAnonymousMethodExpression AnonMethodExpr(Type delegateType)
        {
            IAnonymousMethodExpression iame = AnonMethodExpr();
            iame.DelegateType = TypeRef(delegateType);
            return iame;
        }

        /// <summary>
        /// Creates an array type (of dimenion 1)
        /// </summary>
        /// <param name="type">Element type</param>
        /// <returns>Array type</returns>
        public IArrayType ArrayType(IType type)
        {
            return ArrayType(type, 1);
        }

        /// <summary>
        /// Creates an array type
        /// </summary>
        /// <param name="type">Element type</param>
        /// <param name="rank">Rank of array</param>
        /// <returns>Array type</returns>
        public IArrayType ArrayType(IType type, int rank)
        {
            if (rank < 1) throw new ArgumentException("rank (" + rank + ") < 1");
            IArrayType at = ArrayType();
            at.ElementType = type;
            //for (int i = 0; i < dims; i++) {
            //    at.Dimensions.Add(ArrayDim());
            //}
            // Set up the dotNet type for the array
            if (at is IDotNetType idn)
            {
                Type elementType = ToType(type);
                idn.DotNetType = MakeArrayType(elementType, rank);
            }
            return at;
        }

        /// <summary>
        /// Creates an array expression
        /// </summary>
        /// <param name="type">Element type of array</param>
        /// <param name="sizes">Sizes of each dimension</param>
        /// <returns>Array expression</returns>
        public IArrayCreateExpression ArrayCreateExpr(IType type, params IExpression[] sizes)
        {
            return ArrayCreateExpr(type, (IEnumerable<IExpression>)sizes);
        }

        /// <summary>
        /// Creates an array expression
        /// </summary>
        /// <param name="type">Element type of array</param>
        /// <param name="sizes">Sizes of each dimension</param>
        /// <returns>Array expression</returns>
        public IArrayCreateExpression ArrayCreateExpr(Type type, IEnumerable<IExpression> sizes)
        {
            return ArrayCreateExpr(TypeRef(type), sizes);
        }

        /// <summary>
        /// Creates an array expression
        /// </summary>
        /// <param name="type">Element type of array</param>
        /// <param name="sizes">Sizes of each dimension</param>
        /// <returns>Array expression</returns>
        public IArrayCreateExpression ArrayCreateExpr(IType type, IEnumerable<IExpression> sizes)
        {
            IArrayCreateExpression ace = ArrayCreateExpr();
            ace.Type = type;
            foreach (IExpression size in sizes)
            {
                if (size == null) throw new NullReferenceException("Null size in NewArray()");
                ace.Dimensions.Add(size);
            }
            return ace;
        }

        /// <summary>
        /// Creates an array expression
        /// </summary>
        /// <param name="type">Element type of array</param>
        /// <param name="sizes">Sizes of each dimension</param>
        /// <returns>Array expression</returns>
        public IArrayCreateExpression ArrayCreateExpr(Type type, params IExpression[] sizes)
        {
            return ArrayCreateExpr(TypeRef(type), (IEnumerable<IExpression>)sizes);
        }

        /// <summary>
        /// Creates an array index expression
        /// </summary>
        /// <param name="array">The array expression</param>
        /// <param name="indices">The indices</param>
        /// <returns>Array index expression</returns>
        public IArrayIndexerExpression ArrayIndex(IExpression array, params IExpression[] indices)
        {
            return ArrayIndex(array, (IEnumerable<IExpression>)indices);
        }

        /// <summary>
        /// Creates an array index expression
        /// </summary>
        /// <param name="array">The array expression</param>
        /// <param name="indices">The indices</param>
        /// <returns>Array index expression</returns>
        public IArrayIndexerExpression ArrayIndex(IExpression array, IEnumerable<IExpression> indices)
        {
            foreach (IExpression expr in indices) if (expr == null) throw new NullReferenceException("Array index was null.");
            IArrayIndexerExpression aie = ArrayIndxrExpr();
            aie.Target = array;
            aie.Indices.AddRange(indices);
            return aie;
        }

        /// <summary>
        /// Creates a typeof expression
        /// </summary>
        /// <param name="t">Type</param>
        /// <returns>Typeof expression</returns>
        public ITypeOfExpression TypeOf(IType t)
        {
            ITypeOfExpression toe = TypeOfExpr();
            toe.Type = t;
            return toe;
        }

        /// <summary>
        /// Creates object creation expression
        /// </summary>
        /// <param name="t">Type to create</param>
        /// <param name="pars">Expressions for constructor parameters</param>
        /// <returns>Object create expression</returns>
        public IObjectCreateExpression NewObject(IType t, params IExpression[] pars)
        {
            return NewObject(ToType(t), pars);
        }

        /// <summary>
        /// Creates object creation expression
        /// </summary>
        /// <param name="t">Type to create</param>
        /// <param name="iec">Exprssions for constructor parameters</param>
        /// <returns>Object create expression</returns>
        public IObjectCreateExpression NewObject(IType t, IList<IExpression> iec)
        {
            return NewObject(ToType(t), iec);
        }

        /// <summary>
        /// Creates object creation expression
        /// </summary>
        /// <param name="t">Type to create</param>
        /// <param name="iec">Exprssions for constructor parameters</param>
        /// <returns>Object create expression</returns>
        public IObjectCreateExpression NewObject(Type t, IList<IExpression> iec)
        {
            IExpression[] exps = new IExpression[iec.Count];
            for (int i = 0; i < exps.Length; i++)
                exps[i] = iec[i];
            return NewObject(t, exps);
        }

        /// <summary>
        /// Creates object creation expression
        /// </summary>
        /// <param name="t">Type to create</param>
        /// <param name="args">Expressions for constructor parameters</param>
        /// <returns>Object create expression</returns>
        public IObjectCreateExpression NewObject(Type t, params IExpression[] args)
        {
            if (t.IsArray)
                throw new ArgumentException("Type parameter to NewObject() cannot be an array type.");
            IObjectCreateExpression oce = ObjCreateExpr();
            oce.Type = TypeRef(t);
            if (args != null)
                oce.Arguments.AddRange(args);
            Type[] tp = new Type[args.Length];
            for (int j = 0; j < tp.Length; j++)
                tp[j] = args[j].GetExpressionType();
            oce.Constructor = ConstructorRef(t, tp);
            Microsoft.ML.Probabilistic.Utilities.Assert.IsTrue(oce.Constructor != null);
            return oce;
        }

        /// <summary>
        /// Creates a variable reference expression
        /// </summary>
        /// <param name="ivr">Variable reference</param>
        /// <returns>Variable reference expression</returns>
        public IVariableReferenceExpression VarRefExpr(IVariableReference ivr)
        {
            if (ivr == null) throw new ArgumentNullException(nameof(ivr));
            IVariableReferenceExpression vre = VarRefExpr();
            vre.Variable = ivr;
            return vre;
        }

        /// <summary>
        /// Creates an array of variable reference expressions
        /// </summary>
        /// <param name="ivr">A list of variable declarations</param>
        /// <returns>Array of variable reference expressions</returns>
        public IVariableReferenceExpression[] VarRefExprArray(IList<IVariableDeclaration> ivr)
        {
            IVariableReferenceExpression[] vres = VarRefExprArray(ivr.Count);
            for (int i = 0; i < vres.Length; i++) vres[i] = VarRefExpr(ivr[i]);
            return vres;
        }

        /// <summary>
        /// Creates an assignment expression
        /// </summary>
        /// <param name="target">The target of the assignment</param>
        /// <param name="expr">The expression to assign</param>
        /// <returns>Assignment expression</returns>
        public IAssignExpression AssignExpr(IExpression target, IExpression expr)
        {
            if (target == null) throw new ArgumentNullException(nameof(target));
            if (expr == null) throw new ArgumentNullException(nameof(expr));
            IAssignExpression ae = AssignExpr();
            ae.Target = target;
            ae.Expression = expr;
            return ae;
        }

        /// <summary>
        /// Creates an assignment statement
        /// </summary>
        /// <param name="target">The target of the assignment</param>
        /// <param name="expr">The expression to assign</param>
        /// <returns>Assignment statement</returns>
        public IExpressionStatement AssignStmt(IExpression target, IExpression expr)
        {
            return ExprStatement(AssignExpr(target, expr));
        }

        /// <summary>
        /// Creates a variable declaration
        /// </summary>
        /// <param name="name">Name of variable</param>
        /// <param name="tp">Type of variable</param>
        /// <returns>Variable declaration</returns>
        public IVariableDeclaration VarDecl(string name, Type tp)
        {
            return VarDecl(name, TypeRef(tp));
        }

        /// <summary>
        /// Creates a variable declaration
        /// </summary>
        /// <param name="name">Name of variable</param>
        /// <param name="tp">Type of variable</param>
        /// <returns>Variable declaration</returns>
        public IVariableDeclaration VarDecl(string name, IType tp)
        {
            IVariableDeclaration vd = VarDecl();
            vd.Name = name;
            vd.VariableType = tp;
            return vd;
        }

        /// <summary>
        /// Creates a variable declaration expression
        /// </summary>
        /// <param name="name">Name of variable</param>
        /// <param name="tp">Type of variable</param>
        /// <returns>Variable declaration expression</returns>
        private IVariableDeclarationExpression VarDeclExpr(string name, IType tp)
        {
            return VarDeclExpr(VarDecl(name, tp));
        }

        /// <summary>
        /// Creates a variable declaration expression
        /// </summary>
        /// <param name="ivd">Variable declaration</param>
        /// <returns>Variable declaration expression</returns>
        public IVariableDeclarationExpression VarDeclExpr(IVariableDeclaration ivd)
        {
            IVariableDeclarationExpression vde = VarDeclExpr();
            vde.Variable = ivd;
            return vde;
        }

        /// <summary>
        /// Creates a comment statement
        /// </summary>
        /// <param name="text">Text for the comment statement</param>
        /// <returns>Comment statement</returns>
        public ICommentStatement CommentStmt(string text)
        {
            ICommentStatement cs = CommentStmt();
            IComment c = Comment();
            c.Text = text;
            cs.Comment = c;
            return cs;
        }

        /// <summary>
        /// Character replacements to make valid strings.
        /// </summary>
        public static readonly Dictionary<char, string> replacement = new Dictionary<char, string>();

        static CodeBuilder()
        {
            replacement[' '] = "_";
            replacement['+'] = "Plus";
            replacement['-'] = "Minus";
            replacement['*'] = "Times";
            replacement['/'] = "Over";
            replacement['='] = "Eq";
            replacement['<'] = "Lt";
            replacement['>'] = "Gt";
            replacement['!'] = "Not";
            replacement['&'] = "And";
            replacement['|'] = "Or";
            replacement[','] = "_";
            replacement['~'] = "Tilde";
            replacement['`'] = "Backquote";
            replacement['\''] = "Quote";
            replacement['"'] = "Quote";
            replacement['@'] = "At";
            replacement['#'] = "Num";
            replacement['$'] = "Dollar";
            replacement['%'] = "Percent";
            replacement['^'] = "Caret";
            replacement['\\'] = "Backslash";
            replacement[';'] = "Semicolon";
            replacement[':'] = "Colon";
            replacement['.'] = "Dot";
            replacement['?'] = "Q";
        }

        /// <summary>
        /// Replace characters to make a valid identifier.
        /// </summary>
        /// <param name="name">Identifier</param>
        /// <returns>An alphanumeric string, starting with a letter.</returns>
        public static string MakeValid(string name)
        {
            if (name.Length == 0) return "emptyString";
            StringBuilder s = new StringBuilder();
            bool changed = false;
            if (Char.IsDigit(name[0]))
            {
                s.Append('_');
                changed = true;
            }
            for (int i = 0; i < name.Length; i++)
            {
                char c = name[i];
                if (Char.IsLetterOrDigit(c) || (c == '_'))
                {
                    s.Append(c);
                }
                else if (replacement.ContainsKey(c))
                {
                    s.Append(replacement[c]);
                    changed = true;
                }
                else
                {
                    s.Append('_');
                    changed = true;
                }
            }
            if (changed) return s.ToString();
            else return name;
        }

        /// <summary>
        /// Creates a parameter declaration
        /// </summary>
        /// <param name="name">Name of parameter</param>
        /// <param name="type">Type of parameter</param>
        /// <returns>Parameter declaration</returns>
        public IParameterDeclaration Param(string name, Type type)
        {
            return Param(name, TypeRef(type));
        }

        /// <summary>
        /// Creates a parameter declaration
        /// </summary>
        /// <param name="name">Name of parameter</param>
        /// <param name="type">Type of parameter</param>
        /// <returns>Parameter declaration</returns>
        public IParameterDeclaration Param(string name, IType type)
        {
            IParameterDeclaration pd = ParamDecl();
            pd.Name = name;
            pd.ParameterType = type;
            return pd;
        }

        /// <summary>
        /// Creates a parameter reference expression
        /// </summary>
        /// <param name="pr">Parameter reference</param>
        /// <returns>arameter reference expression</returns>
        public IArgumentReferenceExpression ParamRef(IParameterReference pr)
        {
            IArgumentReferenceExpression are = ParamRef();
            are.Parameter = pr;
            return are;
        }

        /// <summary>
        /// Capitalise a string
        /// </summary>
        /// <param name="s">String to capitalise</param>
        /// <returns>Capitalised string</returns>
        public static string Capitalise(string s)
        {
            if ((s == null) || (s.Length == 0)) return s;
            return Char.ToUpper(s[0], CultureInfo.InvariantCulture) + s.Substring(1);
        }

        /// <summary>
        /// Replaces references to 'ivdFind' with 'ivdReplace' in the supplied expression.
        /// </summary>
        /// <param name="expr">The expression</param>
        /// <param name="ivdFind">Variable declaration to find</param>
        /// <param name="ivdReplace">Replacement variable declaration</param>
        /// <returns>The resulting expression</returns>
        public IExpression ReplaceVariable(IExpression expr, IVariableDeclaration ivdFind, IVariableDeclaration ivdReplace)
        {
            if (expr is IVariableReferenceExpression ivre)
            {
                if (ivre.Variable.Resolve() == ivdFind) return VarRefExpr(ivdReplace);
                return ivre;
            }
            else if (expr is IArrayIndexerExpression iaie)
            {
                IArrayIndexerExpression aie = ArrayIndxrExpr();
                foreach (IExpression ind in iaie.Indices) aie.Indices.Add(ReplaceVariable(ind, ivdFind, ivdReplace));
                aie.Target = ReplaceVariable(iaie.Target, ivdFind, ivdReplace);
                return aie;
            }
            else if (expr is IVariableDeclarationExpression ivde)
            {
                if (ivde.Variable == ivdFind) return VarDeclExpr(ivdReplace);
                return ivde;
            }
            else if (expr is ILiteralExpression) return expr;
            else if (expr is IArgumentReferenceExpression) return expr;
            else if (expr is IPropertyReferenceExpression ipre)
            {
                IExpression target = ReplaceVariable(ipre.Target, ivdFind, ivdReplace);
                if (target == ipre.Target) return ipre;
                IPropertyReferenceExpression pre = PropRefExpr();
                pre.Property = ipre.Property;
                pre.Target = target;
                return pre;
            }
            else throw new NotImplementedException("Unhandled expression type in ReplaceVariable(): " + expr.GetType());
        }

        /// <summary>
        /// Replaces references to 'ivdFind' with specified expression in the supplied expression.
        /// </summary>
        /// <param name="expr">The expression</param>
        /// <param name="ivdFind">Variable declaration to find</param>
        /// <param name="exprReplace">Replacement expression</param>
        /// <param name="replaceCount">Replacement count - passed by reference</param>
        /// <returns>The resulting expression</returns>
        public IExpression ReplaceVariable(IExpression expr, IVariableDeclaration ivdFind, IExpression exprReplace, ref int replaceCount)
        {
            if (expr is IVariableReferenceExpression ivre)
            {
                if (ivre.Variable.Resolve() == ivdFind)
                {
                    replaceCount++;
                    return exprReplace;
                }
                return ivre;
            }
            else if (expr is IArrayIndexerExpression iaie)
            {
                IArrayIndexerExpression aie = ArrayIndxrExpr();
                foreach (IExpression ind in iaie.Indices) aie.Indices.Add(ReplaceVariable(ind, ivdFind, exprReplace, ref replaceCount));
                aie.Target = ReplaceVariable(iaie.Target, ivdFind, exprReplace, ref replaceCount);
                return aie;
            }
            else if (expr is IVariableDeclarationExpression ivde)
            {
                if (ivde.Variable == ivdFind)
                {
                    replaceCount++;
                    return exprReplace;
                }
                return ivde;
            }
            else if (expr is ILiteralExpression) return expr;
            else if (expr is IArgumentReferenceExpression) return expr;
            else if (expr is IPropertyReferenceExpression ipre)
            {
                IExpression target = ReplaceVariable(ipre.Target, ivdFind, exprReplace, ref replaceCount);
                if (target == ipre.Target) return ipre;
                IPropertyReferenceExpression pre = PropRefExpr();
                pre.Property = ipre.Property;
                pre.Target = target;
                return pre;
            }
            else throw new NotImplementedException("Unhandled expression type in ReplaceVariable(): " + expr.GetType());
        }

        /// <summary>
        /// Returns true if the first expression contains the second one.
        /// </summary>
        /// <param name="expr">The expression to search</param>
        /// <param name="exprFind">The expression to look for</param>
        /// <returns>True if the expression was found</returns>
        public bool ContainsExpression(IExpression expr, IExpression exprFind)
        {
            int findCount = 0;
            ReplaceExpression(expr, exprFind, exprFind, ref findCount);
            return (findCount > 0);
        }

        /// <summary>
        /// Finds and replaces one expression with another expression in a given expression
        /// </summary>
        /// <param name="expr">The expression</param>
        /// <param name="exprFind">The expression to be found</param>
        /// <param name="exprReplace">Replacement expression</param>
        /// <returns></returns>
        public IExpression ReplaceExpression(IExpression expr, IExpression exprFind, IExpression exprReplace)
        {
            int dummy = 0;
            return ReplaceExpression(expr, exprFind, exprReplace, ref dummy);
        }

        /// <summary>
        /// Finds and replaces one expression with another expression in a given expression
        /// </summary>
        /// <param name="expr">The expression</param>
        /// <param name="exprFind">The expression to be found</param>
        /// <param name="exprReplace">Replacement expression</param>
        /// <param name="replaceCount">Replacement count - passed by reference</param>
        /// <returns>The resulting expression</returns>
        public IExpression ReplaceExpression(IExpression expr, IExpression exprFind, IExpression exprReplace, ref int replaceCount)
        {
            if (expr == null) return expr;
            else if (expr.Equals(exprFind))
            {
                replaceCount++;
                return exprReplace;
            }
            else if (expr is IArrayIndexerExpression iaie)
            {
                IArrayIndexerExpression aie = ArrayIndxrExpr();
                foreach (IExpression ind in iaie.Indices) aie.Indices.Add(ReplaceExpression(ind, exprFind, exprReplace, ref replaceCount));
                aie.Target = ReplaceExpression(iaie.Target, exprFind, exprReplace, ref replaceCount);
                return aie;
            }
            else if (expr is IPropertyIndexerExpression ipie)
            {
                IPropertyIndexerExpression pie = PropIndxrExpr();
                foreach (IExpression ind in ipie.Indices) pie.Indices.Add(ReplaceExpression(ind, exprFind, exprReplace, ref replaceCount));
                pie.Target = (IPropertyReferenceExpression)ReplaceExpression(ipie.Target, exprFind, exprReplace, ref replaceCount);
                return pie;
            }
            else if (expr is ICastExpression ice)
            {
                return CastExpr(ReplaceExpression(ice.Expression, exprFind, exprReplace, ref replaceCount), ice.TargetType);
            }
            else if (expr is ICheckedExpression iche)
            {
                return CheckedExpr(ReplaceExpression(iche.Expression, exprFind, exprReplace, ref replaceCount));
            }
            else if (
                (expr is IVariableDeclarationExpression) ||
                (expr is IVariableReferenceExpression) ||
                (expr is ILiteralExpression) ||
                (expr is IDefaultExpression) ||
                (expr is IArgumentReferenceExpression)) return expr;
            else if (expr is IPropertyReferenceExpression ipre)
            {
                IExpression target = ReplaceExpression(ipre.Target, exprFind, exprReplace, ref replaceCount);
                if (target == ipre.Target) return ipre;
                IPropertyReferenceExpression pre = PropRefExpr();
                pre.Property = ipre.Property;
                pre.Target = target;
                return pre;
            }
            else if (expr is IArrayCreateExpression)
            {
                IArrayCreateExpression iace = expr as IArrayCreateExpression;
                var ace = ArrayCreateExpr();
                ace.Type = iace.Type;
                ace.Initializer = ReplaceExpression(iace.Initializer, exprFind, exprReplace, ref replaceCount) as IBlockExpression;
                foreach (IExpression dim in iace.Dimensions)
                {
                    ace.Dimensions.Add(ReplaceExpression(dim, exprFind, exprReplace, ref replaceCount));
                }
                return ace;
            }
            else if (expr is IBlockExpression ible)
            {
                IBlockExpression be = BlockExpr();
                foreach (IExpression e in ible.Expressions)
                {
                    be.Expressions.Add(ReplaceExpression(e, exprFind, exprReplace, ref replaceCount));
                }
                return be;
            }
            else if (expr is IMethodInvokeExpression imie)
            {
                IMethodInvokeExpression mie = MethodInvkExpr();
                mie.Method = imie.Method;
                foreach (IExpression arg in imie.Arguments)
                {
                    mie.Arguments.Add(ReplaceExpression(arg, exprFind, exprReplace, ref replaceCount));
                }
                return mie;
            }
            else if (expr is IObjectCreateExpression ioce)
            {
                IObjectCreateExpression oce = ObjCreateExpr();
                oce.Constructor = ioce.Constructor;
                oce.Type = ioce.Type;
                foreach (IExpression arg in ioce.Arguments)
                {
                    oce.Arguments.Add(ReplaceExpression(arg, exprFind, exprReplace, ref replaceCount));
                }
                oce.Initializer = (IBlockExpression)ReplaceExpression(ioce.Initializer, exprFind, exprReplace, ref replaceCount);
                return oce;
            }
            else if (expr is IAnonymousMethodExpression iame)
            {
                IAnonymousMethodExpression ame = AnonMethodExpr();
                ame.DelegateType = iame.DelegateType;
                foreach (IParameterDeclaration ipd in iame.Parameters) ame.Parameters.Add(ipd);
                ame.Body = BlockStmt();
                foreach (IStatement ist in iame.Body.Statements)
                {
                    IStatement st = ist;
                    if (ist is IExpressionStatement ies)
                    {
                        st = ExprStatement(ReplaceExpression(ies.Expression, exprFind, exprReplace, ref replaceCount));
                    }
                    else if (ist is IMethodReturnStatement imrs)
                    {
                        st = Return(ReplaceExpression(imrs.Expression, exprFind, exprReplace, ref replaceCount));
                    }
                    ame.Body.Statements.Add(st);
                }
                return ame;
            }
            else if (expr is IUnaryExpression iue)
            {
                IUnaryExpression ue = UnaryExpr();
                ue.Operator = iue.Operator;
                ue.Expression = ReplaceExpression(iue.Expression, exprFind, exprReplace, ref replaceCount);
                return ue;
            }
            else if (expr is IBinaryExpression ibe)
            {
                IBinaryExpression be = BinaryExpr();
                be.Operator = ibe.Operator;
                be.Left = ReplaceExpression(ibe.Left, exprFind, exprReplace, ref replaceCount);
                be.Right = ReplaceExpression(ibe.Right, exprFind, exprReplace, ref replaceCount);
                return be;
            }
            else if (expr is IMethodReferenceExpression imre)
            {
                var target = ReplaceExpression(imre.Target, exprFind, exprReplace, ref replaceCount);
                return MethodRefExpr(imre.Method, target);
            }
            else if (expr is IThisReferenceExpression)
            {
                return expr;
            }
            else if (expr is IAddressOutExpression iaoe)
            {
                IAddressOutExpression aoe = AddrOutExpr();
                aoe.Expression = ReplaceExpression(iaoe.Expression, exprFind, exprReplace, ref replaceCount);
                return aoe;
            }
            else throw new NotImplementedException("Unhandled expression type in ReplaceExpression(): " + expr.GetType());
        }

        /// <summary>
        /// Creates a nested for loop statement
        /// </summary>
        /// <param name="indexVars">Index variable declarations</param>
        /// <param name="sizes">Size expressions for each dimension</param>
        /// <param name="innerForStatement">Inner for loop statement (output)</param>
        /// <returns>Nested for loop statement</returns>
        public IForStatement NestedForStmt(
            IReadOnlyList<IVariableDeclaration> indexVars,
            IReadOnlyList<IExpression> sizes,
            out IForStatement innerForStatement)
        {
            IForStatement ofs = null;
            IForStatement outerfs = null;
            for (int i = 0; i < sizes.Count; i++)
            {
                IForStatement fs = ForStmt(indexVars[i], sizes[i]);
                if (i == 0) outerfs = fs;
                if (ofs != null) ofs.Body.Statements.Add(fs);
                ofs = fs;
            }
            innerForStatement = ofs;
            return outerfs;
        }

        /// <summary>
        /// Creates a for loop statement
        /// </summary>
        /// <param name="vd">Index variable declaration</param>
        /// <param name="size">Loop size</param>
        /// <returns>For loop statement</returns>
        public IForStatement ForStmt(
            IVariableDeclaration vd,
            IExpression size)
        {
            return ForStmt(vd, LiteralExpr(0), size);
        }

        /// <summary>
        /// Creates a for loop statement
        /// </summary>
        /// <param name="vd">Index variable declaration</param>
        /// <param name="size">Loop size</param>
        /// <param name="start">Start index</param>
        /// <returns>For loop statement</returns>
        public IForStatement ForStmt(
            IVariableDeclaration vd,
            IExpression start,
            IExpression size)
        {
            IVariableDeclarationExpression vde = VarDeclExpr(vd);
            IForStatement fs = ForStmt();
            fs.Initializer = AssignStmt(vde, start);
            IUnaryExpression ue = UnaryExpr();
            ue.Expression = VarRefExpr(vde.Variable);
            ue.Operator = UnaryOperator.PostIncrement;
            fs.Increment = ExprStatement(ue);
            IBinaryExpression be = BinaryExpr();
            be.Left = VarRefExpr(vde.Variable);
            be.Operator = BinaryOperator.LessThan;
            be.Right = size;
            fs.Condition = be;
            fs.Body = BlockStmt();
            return fs;
        }

        public IForStatement ForStmt(IForStatement ifs)
        {
            IForStatement fs = ForStmt();
            if (ifs is BrokenForStatement)
                return new BrokenForStatement(fs);
            else
                return fs;
        }

        /// <summary>
        /// Creates a repeat statement
        /// </summary>
        /// <param name="count">Count i.e. number of repeats</param>
        /// <returns>Repeat statement</returns>
        public IRepeatStatement RepeatStmt(
            IExpression count)
        {
            IRepeatStatement rs = RepeatStmt();
            rs.Count = count;
            rs.Body = BlockStmt();
            return rs;
        }

        /// <summary>
        /// Creates a condition statement.
        /// </summary>
        /// <param name="condition"></param>
        /// <param name="thenBlock"></param>
        /// <returns></returns>
        public IConditionStatement CondStmt(
            IExpression condition,
            IBlockStatement thenBlock
            )
        {
            IConditionStatement ics = CondStmt();
            ics.Condition = condition;
            ics.Then = thenBlock;
            return ics;
        }

        /// <summary>
        /// Creates a condition statement.
        /// </summary>
        /// <param name="condition"></param>
        /// <param name="thenBlock"></param>
        /// <param name="elseBlock"></param>
        /// <returns></returns>
        public IConditionStatement CondStmt(
            IExpression condition,
            IBlockStatement thenBlock,
            IBlockStatement elseBlock
            )
        {
            IConditionStatement ics = CondStmt();
            ics.Condition = condition;
            ics.Then = thenBlock;
            ics.Else = elseBlock;
            return ics;
        }

        /// <summary>
        /// Creates a property reference expression
        /// </summary>
        /// <param name="target">Instance expression</param>
        /// <param name="pr">Declaration of the property</param>
        /// <returns>Property reference expression</returns>
        public IPropertyReferenceExpression PropRefExpr(IExpression target, IPropertyReference pr)
        {
            IPropertyReferenceExpression pre = PropRefExpr();
            pre.Target = target;
            pre.Property = pr;
            return pre;
        }

        /// <summary>
        /// Creates a property reference expression
        /// </summary>
        /// <param name="expr">Instance expression</param>
        /// <param name="declaringType">Declaring type</param>
        /// <param name="propName">Property name</param>
        /// <param name="propType">Property type</param>
        /// <returns>Property reference expression</returns>
        public IPropertyReferenceExpression PropRefExpr(
            IExpression expr,
            Type declaringType,
            string propName,
            Type propType)
        {
            IPropertyReferenceExpression pre = PropRefExpr();
            pre.Target = expr;
            IPropertyReference pr = PropRef();
            pr.DeclaringType = TypeRef(declaringType);
            pr.Name = propName;
            pr.PropertyType = TypeRef(propType);
            pre.Property = pr;
            return pre;
        }


        /// <summary>
        /// Creates a property reference expression
        /// </summary>
        /// <param name="expr">Instance expression</param>
        /// <param name="declaringType">Declaring type</param>
        /// <param name="propName">Property name</param>
        /// <returns>Property reference expression</returns>
        public IPropertyReferenceExpression PropRefExpr(
            IExpression expr,
            Type declaringType,
            string propName)
        {
            Type pt = declaringType.GetProperty(propName).PropertyType;
            return PropRefExpr(expr, declaringType, propName, pt);
        }

        public IPropertyReferenceExpression StaticPropRefExpr(
            Type declaringType,
            string propName)
        {
            return PropRefExpr(TypeRefExpr(declaringType), declaringType, propName);
        }



        /// <summary>
        /// Creates a property declaration with no get or set methods
        /// </summary>
        /// <param name="name">Name of the property</param>
        /// <param name="propertyType">Type of the property</param>
        /// <param name="declaringType">Type containing the property</param>
        /// <returns>A property declaration with no get or set methods</returns>
        private IPropertyDeclaration PropDecl(string name, IType propertyType, IType declaringType)
        {
            IPropertyDeclaration ipd = PropDecl();
            ipd.Name = name;
            ipd.PropertyType = propertyType;
            ipd.DeclaringType = declaringType;
            return ipd;
        }

        /// <summary>
        /// Creates a property declaration with an empty get method and no set method
        /// </summary>
        /// <param name="name">Name of the property</param>
        /// <param name="propertyType">Type of the property</param>
        /// <param name="declaringType">Type containing the property</param>
        /// <param name="getMethodVisibility">Visibility of the get method</param>
        /// <returns>The property declaration with an empty get method and no set method</returns>
        public IPropertyDeclaration PropDecl(string name, Type propertyType, IType declaringType, MethodVisibility getMethodVisibility)
        {
            return PropDecl(name, TypeRef(propertyType), declaringType, getMethodVisibility);
        }

        /// <summary>
        /// Creates a property declaration with an empty get method and no set method
        /// </summary>
        /// <param name="name">Name of the property</param>
        /// <param name="propertyType">Type of the property</param>
        /// <param name="declaringType">Type containing the property</param>
        /// <param name="getMethodVisibility">Visibility of the get method</param>
        /// <returns>The property declaration with an empty get method and no set method</returns>
        private IPropertyDeclaration PropDecl(string name, IType propertyType, IType declaringType, MethodVisibility getMethodVisibility)
        {
            IPropertyDeclaration ipd = PropDecl(name, propertyType, declaringType);
            IMethodDeclaration getMethod = MethodDecl(getMethodVisibility, "get_" + name, propertyType, declaringType);
            getMethod.Body = BlockStmt();
            ipd.GetMethod = getMethod;
            return ipd;
        }

        /// <summary>
        /// Creates a property declaration with an empty get method and set method
        /// </summary>
        /// <param name="name">Name of the property</param>
        /// <param name="propertyType">Type of the property</param>
        /// <param name="declaringType">Type containing the property</param>
        /// <param name="getMethodVisibility">Visibility of the get method</param>
        /// <param name="setMethodVisibility">Visibility of the set method</param>
        /// <param name="value">The value passed to the set method</param>
        /// <returns>The property declaration with an empty get method and set method</returns>
        public IPropertyDeclaration PropDecl(string name, Type propertyType, IType declaringType, MethodVisibility getMethodVisibility, MethodVisibility setMethodVisibility,
                                             out IExpression value)
        {
            return PropDecl(name, TypeRef(propertyType), declaringType, getMethodVisibility, setMethodVisibility, out value);
        }

        /// <summary>
        /// Creates a property declaration with an empty get method and set method
        /// </summary>
        /// <param name="name">Name of the property</param>
        /// <param name="propertyType">Type of the property</param>
        /// <param name="declaringType">Type containing the property</param>
        /// <param name="getMethodVisibility">Visibility of the get method</param>
        /// <param name="setMethodVisibility">Visibility of the set method</param>
        /// <param name="value">The value passed to the set method</param>
        /// <returns>The property declaration with an empty get method and set method</returns>
        private IPropertyDeclaration PropDecl(string name, IType propertyType, IType declaringType, MethodVisibility getMethodVisibility, MethodVisibility setMethodVisibility,
                                              out IExpression value)
        {
            IPropertyDeclaration ipd = PropDecl(name, propertyType, declaringType, getMethodVisibility);
            IParameterDeclaration valueParam = Param("value", propertyType);
            IMethodDeclaration setMethod = MethodDecl(setMethodVisibility, "set_" + name, typeof(void), declaringType, valueParam);
            setMethod.Body = BlockStmt();
            ipd.SetMethod = setMethod;
            value = ParamRef(valueParam);
            return ipd;
        }

        /// <summary>
        /// Creates an event declaration
        /// </summary>
        /// <param name="name">Name of the property</param>
        /// <param name="eventType">Type of the event</param>
        /// <param name="declaringType">Declaring type</param>
        /// <returns>An event declaration</returns>
        public IEventDeclaration EventDecl(string name, ITypeReference eventType, IType declaringType)
        {
            IEventDeclaration ied = EventDecl();
            ied.Name = name;
            ied.EventType = eventType;
            ied.DeclaringType = declaringType;
            return ied;
        }

        /// <summary>
        /// Creates an event declaration
        /// </summary>
        /// <param name="name">Name of the property</param>
        /// <param name="declaringType">Type containing the event</param>
        /// <param name="invokeMethod">Method for clients to invoke the event</param>
        /// <returns>An event declaration</returns>
        public IEventDeclaration EventDecl(string name, Type declaringType, IMethodReference invokeMethod)
        {
            return EventDecl(
                name,
                (ITypeReference)TypeRef(declaringType.GetEvent(name).EventHandlerType),
                TypeRef(declaringType));
        }

        /// <summary>
        /// Create a method declaration for a method which allows clients to fire the event
        /// This is required because events can only be fired from the defining class 
        /// </summary>
        /// <param name="vis">Method visibility</param>
        /// <param name="name">Method name</param>
        /// <param name="eventDecl">The event declaration</param>
        /// <returns>The method declaration</returns>
        public IMethodDeclaration FireEventDecl(
            MethodVisibility vis, string name, IEventDeclaration eventDecl)
        {
            IMethodDeclaration md = MethodDecl();
            md.DeclaringType = eventDecl.DeclaringType;
            md.Name = name;
            md.Visibility = vis;

            // Figure out the signature
            Type eventType = eventDecl.EventType.DotNetType;
            MethodInfo method = eventType.GetMethod("Invoke");
            bool isFirst = true;
            foreach (ParameterInfo param in method.GetParameters())
            {
                if (isFirst)
                {
                    isFirst = false;
                    continue;
                }
                IParameterDeclaration ipd = ParamDecl();
                ipd.Name = param.Name;
                ipd.ParameterType = TypeRef(param.ParameterType);
                md.Parameters.Add(ipd);
            }
            md.ReturnType = MethodReturnType(TypeRef(typeof(void)));
            IBlockStatement ibs = BlockStmt();

            ibs.Statements.Add(CommentStmt("Make a temporary copy of the event to avoid a race condition"));
            ibs.Statements.Add(CommentStmt("if the last subscriber unsubscribes immediately after the null check and before the event is raised."));
            IVariableDeclaration handlerDecl = VarDecl("handler", eventDecl.EventType);
            IExpression handler = VarRefExpr(handlerDecl);
            ibs.Statements.Add(AssignStmt(VarDeclExpr(handlerDecl), EventRefExpr(eventDecl)));

            // Construct if statement
            IExpression condExpr = BinaryExpr(handler, BinaryOperator.ValueInequality, LiteralExpr(null));
            IDelegateInvokeExpression thenExpr = DelegateInvokeExpr();
            thenExpr.Target = handler;
            thenExpr.Arguments.Add(ThisRefExpr());
            foreach (IParameterDeclaration ipd in md.Parameters)
                thenExpr.Arguments.Add(ParamRef(ipd));
            IBlockStatement thenBlock = BlockStmt();
            thenBlock.Statements.Add(ExprStatement(thenExpr));
            IConditionStatement ics = CondStmt(condExpr, thenBlock);
            ibs.Statements.Add(ics);
            md.Body = ibs;
            return md;
        }

        /// <summary>
        /// Creates an event reference expression
        /// </summary>
        /// <param name="ed">Event declaration</param>
        /// <returns>Event reference expression</returns>
        public IEventReferenceExpression EventRefExpr(IEventDeclaration ed)
        {
            IEventReferenceExpression ere = EventRefExpr();
            ere.Event = ed;
            ere.Target = ThisRefExpr();
            return ere;
        }

        /// <summary>
        /// Creates an event reference expression
        /// </summary>
        /// <param name="ed">Event declaration</param>
        /// <param name="target">Target</param>
        /// <returns>Event reference expression</returns>
        public IEventReferenceExpression EventRefExpr(IExpression target, IEventDeclaration ed)
        {
            IEventReferenceExpression ere = EventRefExpr();
            ere.Event = ed;
            ere.Target = target;
            return ere;
        }

        /// <summary>
        /// Throw statement
        /// </summary>
        /// <param name="expr">The expression to throw</param>
        /// <returns></returns>
        public IThrowExceptionStatement ThrowStmt(IExpression expr)
        {
            IThrowExceptionStatement ites = ThrowStmt();
            ites.Expression = expr;
            return ites;
        }

        /// <summary>
        /// default(T) expression
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public IDefaultExpression DefaultExpr(IType type)
        {
            IDefaultExpression ide = DefaultExpr();
            ide.Type = type;
            return ide;
        }

        /// <summary>
        /// default(T) expression
        /// </summary>
        /// <param name="type"></param>
        /// <returns></returns>
        public IDefaultExpression DefaultExpr(Type type)
        {
            return DefaultExpr(TypeRef(type));
        }

        /// <summary>
        /// Creates an array type
        /// </summary>
        /// <param name="tp">Element type</param>
        /// <param name="rank">Rank of array</param>
        /// <returns>Array type</returns>
        public static Type MakeArrayType(Type tp, int rank)
        {
            if (rank == 0) return tp;
            else if (rank == 1) return tp.MakeArrayType(); // bizarrely, gives different result to MakeArrayType(1);
            else return tp.MakeArrayType(rank);
        }

        /// <summary>
        /// Creates a jagged array type
        /// </summary>
        /// <param name="elementType">Element type</param>
        /// <param name="sizes">Expressions for sizes of the sub-arrays</param>
        /// <returns>Jagged array type</returns>
        public static Type MakeJaggedArrayType(Type elementType, IList<IExpression[]> sizes)
        {
            Type tp = elementType;
            for (int i = sizes.Count - 1; i >= 0; i--)
            {
                tp = MakeArrayType(tp, sizes[i].Length);
            }
            return tp;
        }

        /// <summary>
        /// Creates a new jagged array, consisting of a declaration and a nested loop allocating the sub-arrays.
        /// </summary>
        public IVariableDeclaration NewJaggedArray(ICollection<IStatement> addTo, IVariableDeclaration decl, IList<IVariableDeclaration[]> indexVars,
                                                   IList<IExpression[]> sizes, int literalIndexingDepth = 0)
        {
            Type arrayType = ToType(decl.VariableType);
            int rank;
            Type elementType = Util.GetElementType(arrayType, out rank);
            if (sizes.Count == 0)
            {
                addTo.Add(ExprStatement(VarDeclExpr(decl)));
                return decl;
            }
            if (!arrayType.IsAssignableFrom(MakeArrayType(elementType, rank)))
                throw new ArgumentException($"sizes.Count > 0 but {decl} cannot be assigned to an array");

            IStatement declSt = AssignStmt(VarDeclExpr(decl), ArrayCreateExpr(elementType, sizes[0]));
            addTo.Add(declSt);
            IExpression expr = VarRefExpr(decl);
            AddInnerLoops(addTo, expr, elementType, indexVars, sizes, literalIndexingDepth, 0);
            return decl;
        }

        // elementType is the type of expr[0]
        private void AddInnerLoops(ICollection<IStatement> addTo, IExpression expr, Type elementType, IList<IVariableDeclaration[]> indexVars, IList<IExpression[]> sizes, int literalIndexingDepth, int i)
        {
            if (i >= sizes.Count - 1)
                return;
            Type arrayType = elementType;
            int rank;
            elementType = Util.GetElementType(arrayType, out rank);
            if (!arrayType.IsAssignableFrom(MakeArrayType(elementType, rank)))
                throw new ArgumentException($"{StringUtil.TypeToString(arrayType)} cannot be assigned to an array");
            if (i == literalIndexingDepth - 1)
            {
                if (sizes[i].Length != 1) throw new Exception("sizes[i].Length != 1");
                int sizeAsInt = (int)((ILiteralExpression)sizes[i][0]).Value;
                for (int j = 0; j < sizeAsInt; j++)
                {
                    var lhs = ArrayIndex(expr, this.LiteralExpr(j));
                    addTo.Add(AssignStmt(lhs, ArrayCreateExpr(elementType, sizes[i + 1])));
                    AddInnerLoops(addTo, lhs, elementType, indexVars, sizes, literalIndexingDepth, i + 1);
                }
            }
            else
            {
                IForStatement innerFor;
                IForStatement fs = NestedForStmt(indexVars[i], sizes[i], out innerFor);
                addTo.Add(fs);
                addTo = innerFor.Body.Statements;
                var lhs = ArrayIndex(expr, VarRefExprArray(indexVars[i]));
                addTo.Add(AssignStmt(lhs, ArrayCreateExpr(elementType, sizes[i + 1])));
                AddInnerLoops(addTo, lhs, elementType, indexVars, sizes, literalIndexingDepth, i + 1);
            }
        }

        /// <summary>
        /// Creates an expression for a jagged array index
        /// </summary>
        /// <param name="expr">Unindexed expression</param>
        /// <param name="indices">Loop variable declarations</param>
        /// <returns>Indexed expression</returns>
        public IExpression JaggedArrayIndex(IExpression expr, IEnumerable<IEnumerable<IExpression>> indices)
        {
            foreach (var bracket in indices)
            {
                expr = ArrayIndex(expr, bracket);
            }
            return expr;
        }

        public IWhileStatement WhileStmt(IExpression condition)
        {
            IWhileStatement iws = WhileStmt();
            iws.Condition = condition;
            iws.Body = BlockStmt();
            return iws;
        }

        public IWhileStatement WhileStmt(IWhileStatement iws)
        {
            if (iws is FusedBlockStatement)
                return FusedBlockStatement(iws.Condition);
            else
                return WhileStmt(iws.Condition);
        }

        public IWhileStatement FusedBlockStatement(IExpression condition)
        {
            IFusedBlockStatement ifbs = new FusedBlockStatement(condition);
            ifbs.Body = BlockStmt();
            return ifbs;
        }

        public IForStatement BrokenForStatement(IForStatement ifs)
        {
            return new BrokenForStatement(ifs);
        }
    }
}