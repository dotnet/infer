// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Probabilistic.Compiler.Transforms;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace Microsoft.ML.Probabilistic.Compiler.Attributes
{
    /// <summary>
    /// Describes a variable in MSL (random, constant, or loop variable)
    /// </summary>
    /// <remarks>
    /// </remarks>
    internal class VariableInformation : ICompilerAttribute
    {
        /// <summary>
        /// Helps build class declarations
        /// </summary>
        private static readonly CodeBuilder Builder = CodeBuilder.Instance;

        /// <summary>
        /// Helps recognize code patterns
        /// </summary>
        private static readonly CodeRecognizer Recognizer = CodeRecognizer.Instance;

        /// <summary>
        /// Stores the lengths that were used to define an array in MSL.
        /// </summary>
        internal IList<IExpression[]> sizes = new List<IExpression[]>();

        /// <summary>
        /// For jagged arrays, the index variables used in the loops corresponding to the above sizes.
        /// </summary>
        /// <remarks>
        /// May contain null elements for indices that are not variables.
        /// </remarks>
        internal IList<IVariableDeclaration[]> indexVars = new List<IVariableDeclaration[]>();

        /// <summary>
        /// True if this is a stochastic variable.  False for constants and loop variables.
        /// </summary>
        public bool IsStochastic;

        /// <summary>
        /// True if this variable needs a backward message.  Can be true when IsStochastic is false.
        /// </summary>
        public bool NeedsMarginalDividedByPrior;

        // Marginal prototype (this is the prototype of an element, if this is an array)
        internal IExpression marginalPrototypeExpression;
        //internal Type marginalType;

        internal readonly object declaration;

        public string Name
        {
            get
            {
                if (declaration is IVariableDeclaration ivd) return ivd.Name;
                else if (declaration is IParameterDeclaration ipd) return ipd.Name;
                else if (declaration is IFieldDeclaration ifd) return ifd.Name;
                else return null;
            }
        }

        public IType VariableType
        {
            get
            {
                return GetVariableType(declaration);
            }
        }

        /// <summary>
        /// A cache of ToType(VariableType)
        /// </summary>
        internal readonly Type varType;

        private Type innermostElementType;
        internal Type InnermostElementType
        {
            get
            {
                if (innermostElementType == null)
                {
                    Type arrayType = varType;
                    for (int bracket = 0; bracket < sizes.Count; bracket++)
                    {
                        Type elementType = Util.GetElementType(arrayType, out int rank);
                        if (!arrayType.IsAssignableFrom(Util.MakeArrayType(elementType, rank))) break;
                        arrayType = elementType;
                    }
                    innermostElementType = arrayType;
                }
                return innermostElementType;
            }
        }

        /// <summary>
        /// The number of the indexing bracket which always appears indexed by literals, or 0 if none
        /// </summary>
        public int LiteralIndexingDepth;

        /// <summary>
        /// The depth at which to start using distribution arrays in the message type.  Must be at least LiteralIndexingDepth.
        /// </summary>
        public int DistArrayDepth;

        /// <summary>
        /// jagged array depth of varType
        /// </summary>
        private readonly int arrayDepth;

        /// <summary>
        /// Returns the array depth of this variable.  This is the number of pairs of square brackets needed
        /// to fully index the variable i.e. 0 for a non-array, 1 for x[] or x[,], 2 for x[][] or x[,][] or x[,][,] etc. 
        /// </summary>
        public int ArrayDepth
        {
            get { return arrayDepth; }
        }

        public VariableInformation(object declaration)
        {
            this.declaration = declaration;
            varType = Builder.ToType(GetVariableType(declaration));
            var elementType = varType;
            arrayDepth = 0;
            while (elementType.IsArray)
            {
                elementType = elementType.GetElementType();
                arrayDepth++;
            }
        }

        public static IType GetVariableType(object declaration)
        {
            if (declaration is IVariableDeclaration ivd) return ivd.VariableType;
            else if (declaration is IParameterDeclaration ipd) return ipd.ParameterType;
            else if (declaration is IFieldDeclaration ifd) return ifd.FieldType;
            else return null;
        }

        public void SetSizesAtDepth(int depth, IExpression[] lengths)
        {
            if (sizes.Count > depth) throw new NotSupportedException("Attempt to redefine sizes at depth " + depth + ".");
            if (sizes.Count < depth) throw new InferCompilerException("Attempt to set sizes at depth " + depth + " before depth " + (depth - 1) + ".");
            sizes.Add(lengths);
        }

        /// <summary>
        /// Provide missing index variables.
        /// </summary>
        /// <param name="depth">Bracket depth (0 is first bracket)</param>
        /// <param name="vars">May contain null entries for indices that are not variables.</param>
        /// <param name="allowMismatch"></param>
        public void SetIndexVariablesAtDepth(int depth, IVariableDeclaration[] vars, bool allowMismatch = false)
        {
            if (indexVars.Count > depth)
            {
                for (int i = 0; i < vars.Length; i++)
                {
                    if (vars[i] != null)
                    {
                        if (indexVars[depth][i] == null)
                        {
                            indexVars[depth][i] = vars[i];
                        }
                        else if (vars[i] != indexVars[depth][i] && !allowMismatch)
                        {
                            throw new ArgumentException("Invalid definition of array '" + this.Name + "'. Variable '" + vars[i].Name +
                                                        "' cannot be used as an index on the left hand side.  Must use '" + indexVars[depth][i].Name + "'.");
                        }
                    }
                }
                return;
            }
            else if (indexVars.Count < depth) throw new InferCompilerException("Attempt to set index var at depth " + depth + " before depth " + (depth - 1) + ".");
            else indexVars.Add(vars);
        }

        public static string GenerateName(BasicTransformContext context, string prefix)
        {
            int ancIndex = context.FindAncestorIndex<ITypeDeclaration>();
            object input = context.GetAncestor(ancIndex);
            NameGenerator ng = context.InputAttributes.Get<NameGenerator>(input);
            if (ng == null)
            {
                ng = new NameGenerator();
                context.InputAttributes.Set(input, ng);
                object output = context.GetOutputForAncestorIndex<object>(ancIndex);
                context.OutputAttributes.Set(output, ng);
            }
            return ng.GenerateName(prefix);
        }

        public static IVariableDeclaration GenerateLoopVar(BasicTransformContext context, string prefix)
        {
            IVariableDeclaration ivd = Builder.VarDecl(GenerateName(context, prefix), typeof(int));
            //VariableInformation.GetVariableInformation(context, ivd);
            return ivd;
        }

        public void DefineAllIndexVars(BasicTransformContext context)
        {
            DefineIndexVarsUpToDepth(context, sizes.Count);
        }

        public void DefineIndexVarsUpToDepth(BasicTransformContext context, int depth)
        {
            for (int d = 0; d < depth; d++)
            {
                for (int i = 0; i < sizes[d].Length; i++)
                {
                    IVariableDeclaration v = (indexVars.Count <= d) ? null : indexVars[d][i];
                    if (v == null)
                    {
                        v = GenerateLoopVar(context, "_iv");
                    }
                    if (indexVars.Count == d) indexVars.Add(new IVariableDeclaration[sizes[d].Length]);
                    indexVars[d][i] = v;
                }
            }
        }

        public List<IList<IExpression>> GetIndexExpressions(BasicTransformContext context, int depth)
        {
            DefineIndexVarsUpToDepth(context, depth);
            List<IList<IExpression>> indexExprs = new List<IList<IExpression>>();
            for (int d = 0; d < depth; d++)
            {
                IList<IExpression> bracketExprs = Builder.ExprCollection();
                for (int i = 0; i < indexVars[d].Length; i++)
                {
                    IVariableDeclaration indexVar = indexVars[d][i];
                    bracketExprs.Add(Builder.VarRefExpr(indexVar));
                }
                indexExprs.Add(bracketExprs);
            }
            return indexExprs;
        }

        public IExpression GetExpression()
        {
            if (declaration is IVariableDeclaration ivd) return Builder.VarRefExpr(ivd);
            else if (declaration is IParameterDeclaration ipd) return Builder.ParamRef(ipd);
            else if (declaration is IFieldDeclaration ifd) return Builder.FieldRefExpr(ifd);
            else throw new Exception();
        }

        public void DefineSizesUpToDepth(BasicTransformContext context, int arrayDepth)
        {
            IExpression sourceArray = GetExpression();
            for (int depth = 0; depth < arrayDepth; depth++)
            {
                bool notLast = (depth < arrayDepth - 1);
                Type arrayType = sourceArray.GetExpressionType();
                Util.GetElementType(arrayType, out int rank);
                if (sizes.Count <= depth) sizes.Add(new IExpression[rank]);
                IExpression[] indices = new IExpression[rank];
                for (int i = 0; i < rank; i++)
                {
                    if (sizes.Count <= depth || sizes[depth][i] == null)
                    {
                        if (rank == 1)
                        {
                            sizes[depth][i] = Builder.PropRefExpr(sourceArray, arrayType, arrayType.IsArray ? "Length" : "Count", typeof(int));
                        }
                        else
                        {
                            sizes[depth][i] = Builder.Method(sourceArray, typeof(Array).GetMethod("GetLength"), Builder.LiteralExpr(i));
                        }
                    }
                    if (notLast)
                    {
                        if (indexVars.Count <= depth) indexVars.Add(new IVariableDeclaration[rank]);
                        IVariableDeclaration v = indexVars[depth][i];
                        if (v == null)
                        {
                            v = GenerateLoopVar(context, "_iv");
                            indexVars[depth][i] = v;
                        }
                        indices[i] = Builder.VarRefExpr(v);
                    }
                }
                if (notLast) sourceArray = Builder.ArrayIndex(sourceArray, indices);
            }
        }

        /// <summary>
        /// Gets the VariableInformation attribute of a declaration object, or creates one if it doesn't already exist
        /// </summary>
        internal static VariableInformation GetVariableInformation(BasicTransformContext context, object declaration)
        {
            VariableInformation vi = context.InputAttributes.Get<VariableInformation>(declaration);
            if (vi == null)
            {
                vi = new VariableInformation(declaration);
                context.InputAttributes.Set(declaration, vi);
            }
            return vi;
        }


        public override string ToString()
        {
            StringBuilder s = new StringBuilder();
            string stocString = IsStochastic ? "stoc " : (NeedsMarginalDividedByPrior ? "mdp " : "");
            foreach (IExpression[] lengths in sizes)
            {
                s.Append('[');
                bool notFirst = false;
                foreach (IExpression length in lengths)
                {
                    if (notFirst) s.Append(",");
                    else notFirst = true;
                    if (length != null) s.Append(length);
                }
                s.Append(']');
            }
            string sizesString = s.ToString();
            s = new StringBuilder();
            foreach (IVariableDeclaration[] vars in indexVars)
            {
                s.Append('[');
                bool notFirst = false;
                foreach (IVariableDeclaration v in vars)
                {
                    if (notFirst) s.Append(",");
                    else notFirst = true;
                    if (v != null) s.Append(v.Name);
                    //if (v != null) s.Append("("+System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(v)+")");
                }
                s.Append(']');
            }
            string indexVarsString = s.ToString();
            if (indexVarsString.Length > 0) indexVarsString = $",indexVars={indexVarsString}";
            string literalIndexingString = (LiteralIndexingDepth != 0) ? $",LiteralIndexingDepth={LiteralIndexingDepth}" : "";
            string distArrayDepthString = (DistArrayDepth != 0) ? $",DistArrayDepth={DistArrayDepth}" : "";
            string marginalPrototypeString = (marginalPrototypeExpression == null) ? "" : $",{marginalPrototypeExpression}";
            return "VariableInformation(" + stocString + declaration + sizesString + indexVarsString + distArrayDepthString + literalIndexingString + marginalPrototypeString + ")";
        }


        private bool TrySetMarginalPrototypeAutomatically()
        {
            Type marginalType;
            if (this.InnermostElementType == typeof(double))
            {
                marginalType = typeof(Gaussian);
            }
            else if (this.InnermostElementType == typeof(bool))
            {
                marginalType = typeof(Bernoulli);
            }
            else if (this.InnermostElementType == typeof(string))
            {
                marginalType = typeof(StringDistribution);
            }
            else if (this.InnermostElementType == typeof(char))
            {
                marginalType = typeof(DiscreteChar);
            }
            else
            {
                return false;
            }

            this.marginalPrototypeExpression = Builder.NewObject(marginalType);
            return true;
        }

        /// <summary>
        /// Sets the marginal prototype from a supplied MarginalPrototype attribute.
        /// If this is null, attempts to set the marginal prototype automatically.
        /// </summary>
        /// <param name="mpa"></param>
        /// <param name="throwIfMissing"></param>
        internal bool SetMarginalPrototypeFromAttribute(MarginalPrototype mpa, bool throwIfMissing = true)
        {
            if (mpa != null)
            {
                if (mpa.prototypeExpression != null)
                {
                    marginalPrototypeExpression = mpa.prototypeExpression;
                }
                else
                {
                    marginalPrototypeExpression = Quoter.Quote(mpa.prototype);
                }
            }
            else if (!TrySetMarginalPrototypeAutomatically())
            {
                if (throwIfMissing)
                    throw new ArgumentException("Cannot automatically determine distribution type for variable type '" + StringUtil.TypeToString(InnermostElementType) + "'" +
                                                ": you must specify a ValueRange or MarginalPrototype attribute for variable '" + Name + "' or its parent variables.");
                else
                    return false;
            }
            return true;
        }

        internal IExpression GetMarginalPrototypeExpression(BasicTransformContext context, IExpression prototypeExpression,
            IList<IList<IExpression>> indices, IList<IList<IExpression>> wildcardVars = null)
        {
            IExpression original = prototypeExpression;
            int replaceCount = 0;
            prototypeExpression = ReplaceIndexVars(context, prototypeExpression, indices, wildcardVars, ref replaceCount);
            int mpDepth = Util.GetArrayDepth(varType, Distribution.GetDomainType(prototypeExpression.GetExpressionType()));
            int indexingDepth = indices.Count;
            int wildcardBracket = 0;
            for (int depth = mpDepth; depth < indexingDepth; depth++)
            {
                IList<IExpression> indexCollection = Builder.ExprCollection();
                int wildcardCount = 0;
                for (int i = 0; i < indices[depth].Count; i++)
                {
                    if (Recognizer.IsStaticMethod(indices[depth][i], new Func<int>(GateAnalysisTransform.AnyIndex)))
                    {
                        indexCollection.Add(wildcardVars[wildcardBracket][wildcardCount]);
                        wildcardCount++;
                    }
                    else
                    {
                        indexCollection.Add(indices[depth][i]);
                    }
                }
                if (indexCollection.Count > 0)
                {
                    if (wildcardCount > 0) wildcardBracket++;
                    prototypeExpression = Builder.ArrayIndex(prototypeExpression, indexCollection);
                    replaceCount++;
                }
            }
            if (replaceCount > 0) return prototypeExpression;
            else return original;
        }

        /// <summary>
        /// Create an array of this variable type, optionally after slicing it.
        /// </summary>
        /// <param name="addTo"></param>
        /// <param name="context"></param>
        /// <param name="name">Name of the new variable.</param>
        /// <param name="arraySize">Length of the array.  Cannot be null.</param>
        /// <param name="newIndexVar">Name of a new integer variable used to index the array.  Cannot be null.</param>
        /// <param name="indices">Indices applied to this, before creating the array.  May be null (equivalent to an empty array).  May contain wildcards.</param>
        /// <param name="wildcardVars">Loop variables to use for wildcards.  May be null if there are no wildcards.</param>
        /// <param name="useLiteralIndices">If true, literal indices will be used instead of newIndexVar.</param>
        /// <param name="copyInitializer">If true, the new variable will be given an InitialiseTo attribute if this variable had one</param>
        /// <param name="useArrays">If true, the message type of the array will be a .NET array instead of DistributionArray.  Implied by <paramref name="useLiteralIndices"/>.</param>
        /// <remarks>
        /// The new array is indexed by wildcards first, then newIndexVar, then the indices remaining from the original array.
        /// For example, if original array is indexed [i,j][k,l][m,n] and indices = [*,*][3,*] then 
        /// the new array is indexed [wildcard0,wildcard1][wildcard2][newIndexVar][m,n] where sizes and marginalPrototype have expressions replaced according
        /// to (i=wildcard0, j=wildcard1, k=3, l=wildcard2).
        /// </remarks>
        /// <returns>A new array of depth <c>(arraySize != null) + ArrayDepth - indices.Count + wildcardVars.Count</c></returns>
        internal IVariableDeclaration DeriveArrayVariable(ICollection<IStatement> addTo, BasicTransformContext context, string name,
                                                          IExpression arraySize, IVariableDeclaration newIndexVar,
                                                          IList<IList<IExpression>> indices = null,
                                                          IList<IList<IExpression>> wildcardVars = null,
                                                          bool useLiteralIndices = false,
                                                          bool copyInitializer = false,
                                                          bool useArrays = false)
        {
            if (arraySize == null)
                throw new ArgumentException("arraySize is null");
            if (newIndexVar == null)
                throw new ArgumentException("newIndexVar is null");
            return DeriveArrayVariable(addTo, context, name,
                new IExpression[][] { new[] { arraySize } },
                new IVariableDeclaration[][] { new[] { newIndexVar } },
                indices, wildcardVars, useLiteralIndices, copyInitializer, useArrays);
        }

        internal IVariableDeclaration DeriveArrayVariable(ICollection<IStatement> addTo, BasicTransformContext context, string name,
                                                          IList<IExpression[]> arraySize, IList<IVariableDeclaration[]> newIndexVar,
                                                          IList<IList<IExpression>> indices = null,
                                                          IList<IList<IExpression>> wildcardVars = null,
                                                          bool useLiteralIndices = false,
                                                          bool copyInitializer = false,
                                                          bool useArrays = false)
        {
            List<IExpression[]> newSizes = new List<IExpression[]>();
            List<IVariableDeclaration[]> newIndexVars = new List<IVariableDeclaration[]>();
            Type innerType = varType;
            if (indices != null)
            {
                // add wildcard variables to newIndexVars
                for (int i = 0; i < indices.Count; i++)
                {
                    List<IExpression> sizeBracket = new List<IExpression>();
                    List<IVariableDeclaration> indexVarsBracket = new List<IVariableDeclaration>();
                    for (int j = 0; j < indices[i].Count; j++)
                    {
                        IExpression index = indices[i][j];
                        if (Recognizer.IsStaticMethod(index, new Func<int>(GateAnalysisTransform.AnyIndex)))
                        {
                            int replaceCount = 0;
                            sizeBracket.Add(ReplaceIndexVars(context, sizes[i][j], indices, wildcardVars, ref replaceCount));
                            IVariableDeclaration v = indexVars[i][j];
                            if (wildcardVars != null) v = Recognizer.GetVariableDeclaration(wildcardVars[newIndexVars.Count][indexVarsBracket.Count]);
                            else if (Recognizer.GetLoopForVariable(context, v) != null)
                            {
                                // v is already used in a parent loop.  must generate a new variable.
                                v = GenerateLoopVar(context, "_a");
                            }
                            indexVarsBracket.Add(v);
                        }
                    }
                    if (sizeBracket.Count > 0)
                    {
                        newSizes.Add(sizeBracket.ToArray());
                        newIndexVars.Add(indexVarsBracket.ToArray());
                    }

                    innerType = Util.GetElementType(innerType);
                }
            }
            int literalIndexingDepth = 0;
            int distArrayDepth = 0;
            if (arraySize != null)
            {
                newSizes.AddRange(arraySize);
                if (useArrays || useLiteralIndices)
                    distArrayDepth = newSizes.Count;
                if (useLiteralIndices)
                    literalIndexingDepth = newSizes.Count;
                newIndexVars.AddRange(newIndexVar);
            }
            // innerType may not be an array type, so we create the new array type here instead of descending further.
            Type arrayType = CodeBuilder.MakeJaggedArrayType(innerType, newSizes);
            int indexingDepth = (indices == null) ? 0 : indices.Count;
            List<IList<IExpression>> replacements = new List<IList<IExpression>>();
            if (indices != null) replacements.AddRange(indices);
            for (int i = indexingDepth; i < sizes.Count; i++)
            {
                if (replacements.Count == 0)
                {
                    newSizes.Add(sizes[i]);
                    if (indexVars.Count > i) newIndexVars.Add(indexVars[i]);
                }
                else
                {
                    // must substitute references to indexVars with indices
                    IExpression[] sizeBracket = new IExpression[sizes[i].Length];
                    IVariableDeclaration[] indexVarBracket = new IVariableDeclaration[sizes[i].Length];
                    IList<IExpression> replacementBracket = Builder.ExprCollection();
                    for (int j = 0; j < sizeBracket.Length; j++)
                    {
                        int replaceCount = 0;
                        sizeBracket[j] = ReplaceIndexVars(context, sizes[i][j], replacements, wildcardVars, ref replaceCount);
                        if (replaceCount > 0) indexVarBracket[j] = GenerateLoopVar(context, "_a");
                        else if (indexVars.Count > i) indexVarBracket[j] = indexVars[i][j];
                        if (indexVarBracket[j] != null) replacementBracket.Add(Builder.VarRefExpr(indexVarBracket[j]));
                    }
                    newSizes.Add(sizeBracket);
                    newIndexVars.Add(indexVarBracket);
                    replacements.Add(replacementBracket);
                }
            }

            IVariableDeclaration arrayvd = Builder.VarDecl(CodeBuilder.MakeValid(name), arrayType);
            Builder.NewJaggedArray(addTo, arrayvd, newIndexVars, newSizes, literalIndexingDepth);
            context.InputAttributes.CopyObjectAttributesTo(declaration, context.OutputAttributes, arrayvd);
            // cannot copy the initializer since it will have a different size.
            context.OutputAttributes.Remove<InitialiseTo>(arrayvd);
            context.OutputAttributes.Remove<InitialiseBackwardTo>(arrayvd);
            context.OutputAttributes.Remove<InitialiseBackward>(arrayvd);
            context.OutputAttributes.Remove<VariableInformation>(arrayvd);
            context.OutputAttributes.Remove<SuppressVariableFactor>(arrayvd);
            context.OutputAttributes.Remove<LoopContext>(arrayvd);
            context.OutputAttributes.Remove<Containers>(arrayvd);
            context.OutputAttributes.Remove<ChannelInfo>(arrayvd);
            context.OutputAttributes.Remove<IsInferred>(arrayvd);
            context.OutputAttributes.Remove<QueryTypeCompilerAttribute>(arrayvd);
            context.OutputAttributes.Remove<DerivMessage>(arrayvd);
            context.OutputAttributes.Remove<PointEstimate>(arrayvd);
            context.OutputAttributes.Remove<DescriptionAttribute>(arrayvd);
            context.OutputAttributes.Remove<MarginalPrototype>(arrayvd);
            VariableInformation vi = VariableInformation.GetVariableInformation(context, arrayvd);
            vi.IsStochastic = IsStochastic;
            vi.NeedsMarginalDividedByPrior = NeedsMarginalDividedByPrior;
            vi.sizes = newSizes;
            vi.indexVars = newIndexVars;
            vi.DistArrayDepth = distArrayDepth + System.Math.Max(0, this.DistArrayDepth - indexingDepth);
            vi.LiteralIndexingDepth = literalIndexingDepth + System.Math.Max(0, this.LiteralIndexingDepth - indexingDepth);
            if (marginalPrototypeExpression != null)
            {
                // substitute indices in the marginal prototype expression
                vi.marginalPrototypeExpression = GetMarginalPrototypeExpression(context, marginalPrototypeExpression, replacements, wildcardVars);
            }
            InitialiseTo it = context.InputAttributes.Get<InitialiseTo>(declaration);
            if (it != null && copyInitializer)
            {
                // if original array is indexed [i,j][k,l][m,n] and indices = [*,*][3,*] then
                // initExpr2 = new PlaceHolder[wildcard0,wildcard1] { new PlaceHolder[wildcard2] { new PlaceHolder[newIndexVar] { initExpr[wildcard0,wildcard1][3,wildcard2] } } }
                IExpression initExpr = it.initialMessagesExpression;
                // add indices to the initialiser expression
                int wildcardBracket = 0;
                for (int depth = 0; depth < indexingDepth; depth++)
                {
                    IList<IExpression> indexCollection = Builder.ExprCollection();
                    int wildcardCount = 0;
                    for (int i = 0; i < indices[depth].Count; i++)
                    {
                        if (Recognizer.IsStaticMethod(indices[depth][i], new Func<int>(GateAnalysisTransform.AnyIndex)))
                        {
                            indexCollection.Add(wildcardVars[wildcardBracket][wildcardCount]);
                            wildcardCount++;
                        }
                        else
                        {
                            indexCollection.Add(indices[depth][i]);
                        }
                    }
                    if (indexCollection.Count > 0)
                    {
                        if (wildcardCount > 0) wildcardBracket++;
                        initExpr = Builder.ArrayIndex(initExpr, indexCollection);
                    }
                }
                // add array creates to the initialiser expression
                if (newIndexVar != null)
                {
                    initExpr = MakePlaceHolderArrayCreate(initExpr, newIndexVar);
                }
                if (wildcardBracket > 0)
                {
                    while (wildcardBracket > 0)
                    {
                        wildcardBracket--;
                        initExpr = MakePlaceHolderArrayCreate(initExpr, vi.indexVars[wildcardBracket]);
                    }
                }
                context.OutputAttributes.Set(arrayvd, new InitialiseTo(initExpr));
            }
            ChannelTransform.setAllGroupRoots(context, arrayvd, false);
            return arrayvd;
        }

        internal static IExpression MakePlaceHolderArrayCreate(IExpression expr, IList<IVariableDeclaration[]> indexVars)
        {
            for (int bracket = indexVars.Count - 1; bracket >= 0; bracket--)
            {
                expr = MakePlaceHolderArrayCreate(expr, indexVars[bracket]);
            }
            return expr;
        }

        internal static IExpression MakePlaceHolderArrayCreate(IExpression expr, IList<IVariableDeclaration> indexVars)
        {
            CodeBuilder Builder = CodeBuilder.Instance;
            IArrayCreateExpression iace = Builder.ArrayCreateExpr(typeof(PlaceHolder), Util.ArrayInit(indexVars.Count, i => Builder.VarRefExpr(indexVars[i])));
            iace.Initializer = Builder.BlockExpr();
            iace.Initializer.Expressions.Add(expr);
            return iace;
        }

        /// <summary>
        /// Create a slice of this variable array, where all indices up to a certain depth are given.
        /// </summary>
        /// <param name="addTo"></param>
        /// <param name="context"></param>
        /// <param name="name">Name of the new variable array</param>
        /// <param name="indices">Expressions used to index the variable array.  May contain wildcards.</param>
        /// <param name="wildcardVars">Loop variables to use for wildcards.  May be null if there are no wildcards.</param>
        /// <param name="copyInitializer">If true, the new variable will be given an InitialiseTo attribute if this variable had one</param>
        /// <returns>The declaration of the new variable.</returns>
        /// <remarks>
        /// For example, suppose we want to slice a[i][2][j][k] into b[j][k].
        /// Then <paramref name="name"/>="b", <paramref name="indices"/>=<c>[i][2]</c>.
        /// </remarks>
        internal IVariableDeclaration DeriveIndexedVariable(IList<IStatement> addTo, BasicTransformContext context, string name,
                                                            List<IList<IExpression>> indices = null, IList<IList<IExpression>> wildcardVars = null,
                                                            bool copyInitializer = false)
        {
            return DeriveArrayVariable(addTo, context, name, (IList<IExpression[]>)null, null, indices, wildcardVars, copyInitializer: copyInitializer);
        }

        /// <summary>
        /// Replace all indexVars which appear in expr with the given indices.
        /// </summary>
        /// <param name="context"></param>
        /// <param name="expr">Any expression</param>
        /// <param name="indices">A list of lists of index expressions (one list for each indexing bracket).</param>
        /// <param name="wildcardIndices">Expressions used to replace wildcards.  May be null if there are no wildcards.</param>
        /// <param name="replaceCount">Incremented for each replacement.</param>
        /// <returns>A new expression.</returns>
        internal IExpression ReplaceIndexVars(BasicTransformContext context, IExpression expr, IList<IList<IExpression>> indices,
                                              IList<IList<IExpression>> wildcardIndices, ref int replaceCount)
        {
            Dictionary<IVariableDeclaration, IExpression> replacedIndexVars = new Dictionary<IVariableDeclaration, IExpression>();
            int wildcardBracket = 0;
            for (int depth = 0; depth < indices.Count; depth++)
            {
                if (indexVars.Count > depth)
                {
                    int wildcardCount = 0;
                    for (int i = 0; i < indices[depth].Count; i++)
                    {
                        if (indexVars[depth].Length > i)
                        {
                            IVariableDeclaration indexVar = indexVars[depth][i];
                            if (indexVar != null)
                            {
                                IExpression actualIndex = indices[depth][i];
                                if (Recognizer.IsStaticMethod(actualIndex, new Func<int>(GateAnalysisTransform.AnyIndex)))
                                {
                                    actualIndex = wildcardIndices[wildcardBracket][wildcardCount];
                                    wildcardCount++;
                                }
                                IExpression formalIndex = Builder.VarRefExpr(indexVar);
                                if (!formalIndex.Equals(actualIndex))
                                {
                                    expr = Builder.ReplaceExpression(expr, formalIndex, actualIndex, ref replaceCount);
                                    replacedIndexVars.Add(indexVar, actualIndex);
                                }
                            }
                        }
                    }
                    if (wildcardCount > 0) wildcardBracket++;
                }
            }
            CheckReplacements(context, expr, replacedIndexVars);
            return expr;
        }

        /// <summary>
        /// Check that the replacements are safe.  
        /// </summary>
        /// <param name="context"></param>
        /// <param name="expr"></param>
        /// <param name="replacedIndexVars"></param>
        private static void CheckReplacements(BasicTransformContext context, IExpression expr, Dictionary<IVariableDeclaration, IExpression> replacedIndexVars)
        {
            foreach (var v in Recognizer.GetVariables(expr))
            {
                Containers containers = context.InputAttributes.Get<Containers>(v);
                if (containers != null && !replacedIndexVars.ContainsKey(v))
                {
                    foreach (IStatement container in containers.inputs)
                    {
                        if (container is IForStatement ifs)
                        {
                            IVariableDeclaration loopVar = Recognizer.LoopVariable(ifs);
                            if (replacedIndexVars.TryGetValue(loopVar, out IExpression actualIndex))
                            {
                                context.Error($"Cannot index {expr} by {loopVar.Name}={actualIndex} since {v.Name} has an implicit dependency on {loopVar.Name}. Try making the dependency explicit by putting {v.Name} into an array indexed by {loopVar.Name}");
                            }
                        }
                    }
                }
            }
        }

        internal bool HasIndexVar(IVariableDeclaration ivd)
        {
            foreach (IVariableDeclaration[] bracket in indexVars)
            {
                foreach (IVariableDeclaration indexVar in bracket)
                {
                    if (indexVar == null) continue;
                    if (indexVar.Name == ivd.Name)
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        internal List<IStatement> BuildWildcardLoops(IList<IList<IExpression>> wildcardVars)
        {
            List<IStatement> loops = new List<IStatement>();
            for (int i = 0; i < wildcardVars.Count; i++)
            {
                for (int j = 0; j < wildcardVars[i].Count; j++)
                {
                    IExpression size = sizes[i][j];
                    IVariableDeclaration v = Recognizer.GetVariableDeclaration(wildcardVars[i][j]);
                    loops.Add(Builder.ForStmt(v, size));
                }
            }
            return loops;
        }

        internal bool IsPartitionedAtDepth(BasicTransformContext context, int depth)
        {
            if (depth >= indexVars.Count) return false;
            IVariableDeclaration[] bracket = indexVars[depth];
            bool allPartitioned = true;
            bool anyPartitioned = false;
            for (int i = 0; i < bracket.Length; i++)
            {
                IVariableDeclaration indexVar = bracket[i];
                bool isPartitioned = (indexVar != null && context.InputAttributes.Has<Partitioned>(indexVar));
                if (isPartitioned) anyPartitioned = true;
                else allPartitioned = false;
            }
            if (allPartitioned) return true;
            else if (anyPartitioned) throw new Exception("indexing bracket is partially partitioned");
            else return false;
        }
    }

    internal class NameGenerator : ICompilerAttribute
    {
        private readonly Dictionary<string, int> counts = new Dictionary<string, int>();

        public string GenerateName(string prefix)
        {
            if (prefix.Length > 0)
            {
                // If prefix ends with a digit, append an underscore.
                // This ensures that names generated from different prefixes cannot collide.
                char lastChar = prefix[prefix.Length - 1];
                if (char.IsDigit(lastChar))
                    prefix += "_";
            }
            counts.TryGetValue(prefix, out int count);
            if (count == 0) count = 1;
            counts[prefix] = count + 1;
            if (count == 1) return prefix;
            return prefix + count;
        }
    }
}