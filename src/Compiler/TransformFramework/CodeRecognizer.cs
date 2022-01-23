// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler.Transforms;
using System.Linq;
using Microsoft.ML.Probabilistic.Factors;

namespace Microsoft.ML.Probabilistic.Compiler
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// Helper class for recognizing patterns in code which is to be transformed.
    /// </summary>
    public class CodeRecognizer
    {
        public static readonly CodeRecognizer Instance = new CodeRecognizer();

        private CodeRecognizer()
        {
        }

        /// <summary>
        /// Helps build class declarations
        /// </summary>
        private static readonly CodeBuilder Builder = CodeBuilder.Instance;

        /// <summary>
        /// True if expr is a MethodInvokeExpression on any static method of type.
        /// </summary>
        /// <param name="expr"></param>
        /// <param name="type"></param>
        /// <returns></returns>
        public bool IsStaticMethod(IExpression expr, Type type)
        {
            return (GetStaticMethodOfType(expr, type) != null);
        }

        /// <summary>
        /// True if expr is a MethodInvokeExpression on any overload of the named static method of type.
        /// </summary>
        /// <param name="expr"></param>
        /// <param name="type"></param>
        /// <param name="methodName"></param>
        /// <returns></returns>
        public bool IsStaticMethod(IExpression expr, Type type, string methodName)
        {
            string s = GetStaticMethodOfType(expr, type);
            return methodName.Equals(s);
        }

        public bool IsStaticMethod(IExpression expr, Delegate method)
        {
            return IsStaticMethod(expr, method.Method);
        }

        public bool IsStaticMethod(IExpression expr, MethodInfo method)
        {
            IMethodReference imr = GetMethodReference(expr);
            if (imr == null)
                return false;
            return method.Equals(imr.MethodInfo);
        }

        /// <summary>
        /// True if imie is a MethodInvokeExpression on any overload of the named static generic method of type.
        /// </summary>
        /// <param name="expr"></param>
        /// <param name="type"></param>
        /// <param name="methodName"></param>
        /// <returns></returns>
        public bool IsStaticGenericMethod(IExpression expr, Type type, string methodName)
        {
            if (!(expr is IMethodInvokeExpression imie))
                return false;
            if (!(imie.Method is IMethodReferenceExpression imre))
                return false;
            if (!(imre.Target is ITypeReferenceExpression itre))
                return false;
            ITypeReference itr = itre.Type;
            if (itr.Namespace != type.Namespace)
                return false;
            if (itr.GenericType is ITypeReference)
                itr = itr.GenericType;
            if (itr.DotNetType.Name != type.Name)
                return false;
            return (imie.Method.Method.Name == methodName);
        }

        public bool IsStaticGenericMethod(IExpression expr, Delegate method)
        {
            MethodInfo methodInfo = method.Method;
            return IsStaticGenericMethod(expr, methodInfo.DeclaringType, methodInfo.Name);
        }

        public string GetStaticMethodOfType(IExpression expr, Type type)
        {
            if (!(expr is IMethodInvokeExpression imie))
                return null;
            if (!(imie.Method is IMethodReferenceExpression imre))
                return null;
            if (!(imre.Target is ITypeReferenceExpression itre))
                return null;
            if (!IsTypeReferenceTo(itre, type))
                return null;
            return imre.Method.Name;
        }

        public ITypeReference GetStaticMethodType(IExpression expr)
        {
            if (!(expr is IMethodInvokeExpression imie))
                return null;
            if (!(imie.Method is IMethodReferenceExpression imre))
                return null;
            if (!(imre.Target is ITypeReferenceExpression itre))
                return null;
            return itre.Type;
        }

        public IMethodReference GetMethodReference(IExpression expr)
        {
            if (!(expr is IMethodInvokeExpression imie))
                return null;
            if (!(imie.Method is IMethodReferenceExpression imre))
                return null;
            return imre.Method;
        }

        public bool IsBeingDeclared(BasicTransformContext context)
        {
            object parent = context.GetAncestor(context.InputStack.Count - 2);
            return (parent is IVariableDeclarationExpression);
        }

        public bool IsBeingIndexed(BasicTransformContext context)
        {
            object parent = context.GetAncestor(context.InputStack.Count - 2);
            return (parent is IArrayIndexerExpression);
        }

        /// <summary>
        /// Returns true if the expression is being mutated in the given context.  For example, this will
        /// be true if the expression is the LHS of an assignment, or the argument to an 'out' parameter.
        /// </summary>
        /// <param name="context"></param>
        /// <param name="expr"></param>
        /// <returns></returns>
        public bool IsBeingMutated(BasicTransformContext context, IExpression expr)
        {
            int methodIndex = context.FindAncestorIndex<IMethodInvokeExpression>();
            if (methodIndex != -1)
            {
                if (context.GetAncestor(methodIndex + 1) is IAddressOutExpression iaoe)
                    return IsPartOf(iaoe.Expression, expr);
                else
                    return false;
            }
            return IsOnLHSOfAssignment(context, expr);
        }

        public Bounds GetBounds(IExpression expr, IReadOnlyDictionary<IVariableDeclaration, Bounds> bounds)
        {
            if (expr is IVariableReferenceExpression && bounds != null)
            {
                IVariableDeclaration ivd = GetVariableDeclaration(expr);
                if (bounds.TryGetValue(ivd, out Bounds b))
                    return b;
            }
            else if (expr is IBinaryExpression ibe)
            {
                Bounds left = GetBounds(ibe.Left, bounds);
                Bounds right = GetBounds(ibe.Right, bounds);
                if (left != null && right != null)
                {
                    if (ibe.Operator == BinaryOperator.Add)
                    {
                        return new Bounds()
                        {
                            lowerBound = left.lowerBound + right.lowerBound,
                            upperBound = left.upperBound + right.upperBound
                        };
                    }
                    else if (ibe.Operator == BinaryOperator.Subtract)
                    {
                        return new Bounds()
                        {
                            lowerBound = left.lowerBound - right.upperBound,
                            upperBound = left.upperBound - right.lowerBound
                        };
                    }
                }
            }
            else if (expr is ILiteralExpression ile)
            {
                if (ile.Value is int value)
                {
                    return new Bounds()
                    {
                        lowerBound = value,
                        upperBound = value
                    };
                }
            }
            return null;
        }

        private IVariableDeclaration GetOffsetVariable(IExpression expr)
        {
            return GetOffsetVariable(expr, out int offset);
        }

        private IVariableDeclaration GetOffsetVariable(IExpression expr, out int offset)
        {
            offset = 0;
            if (expr is IVariableReferenceExpression)
            {
                return GetVariableDeclaration(expr);
            }
            else if (expr is IBinaryExpression indexBinaryExpr)
            {
                if (indexBinaryExpr.Left is IVariableReferenceExpression ivre && 
                    indexBinaryExpr.Right is ILiteralExpression offsetExpr && 
                    offsetExpr.Value is int)
                {
                    offset = (int)offsetExpr.Value;
                    if (indexBinaryExpr.Operator == BinaryOperator.Subtract)
                    {
                        offset = -offset;
                    }
                    else if (indexBinaryExpr.Operator == BinaryOperator.Add)
                    {
                        // do nothing
                    }
                    else
                    {
                        return null;
                    }
                    return GetVariableDeclaration(ivre);
                }
            }
            return null;
        }

        /// <summary>
        /// Reflection cache
        /// </summary>
        readonly MethodInfo AnyIndexMethod = new Func<int>(GateAnalysisTransform.AnyIndex).Method;

        // helper for MutatingFirstAffectsSecond
        // offsets and extraIndices only need to be modified on a match (though they can be modified in any case)
        private bool IndicesOverlap(IList<IExpression> mutated_indices, IList<IExpression> affected_indices, bool mutatesWithinOnly,
                                    IReadOnlyDictionary<IVariableDeclaration, Bounds> boundsInMutated,
                                    IReadOnlyDictionary<IVariableDeclaration, Bounds> boundsInAffected,
                                    OffsetInfo offsets,
                                    ICollection<IVariableDeclaration> extraIndices,
                                    ICollection<IVariableDeclaration> matchedIndices)
        {
            // if mutatesWithinOnly = false, then return false on mismatched literals
            // if mutatesWithinOnly = true, then return false also if mutated index is wildcard and affected index is literal
            // if one index is an offset of the other and non-literal, return true but add entry to offsets dictionary
            if (mutated_indices.Count != affected_indices.Count)
                throw new ArgumentException("Mismatched array rank");
            for (int i = 0; i < mutated_indices.Count; i++)
            {
                IExpression mutated_index = mutated_indices[i];
                IExpression affected_index = affected_indices[i];
                if (mutated_index.Equals(affected_index))
                {
                    IVariableDeclaration mutatedVar = GetOffsetVariable(mutated_index);
                    if (mutatedVar != null) matchedIndices.Add(mutatedVar);
                    continue;
                }
                if (IsStaticMethod(affected_index, AnyIndexMethod))
                    continue;
                if (affected_index is ILiteralExpression affected_literal)
                {
                    if (mutated_index is ILiteralExpression)
                    {
                        // if we reach this point, they don't match
                        return false;
                    }
                    else
                    {
                        if (mutatesWithinOnly)
                            return false; // mutated_index is more general
                        int affected_value = (int)affected_literal.Value;
                        Bounds mutatedBounds = GetBounds(mutated_index, boundsInMutated);
                        if (mutatedBounds != null)
                        {
                            if (!mutatedBounds.Contains(affected_value))
                                return false;
                        }
                        if (offsets != null)
                        {
                            // if affected=v (when i=j) and mutated=i-k, then we write to v on iter v+k, and read on iter j, so offset is v+k-j
                            // offset = (loop counter of write) - (loop counter of read)
                            Bounds affectedBounds = GetBounds(mutated_index, boundsInAffected);
                            if (affectedBounds != null)
                            {
                                // affectedBounds = [jLower-k, jUpper-k]
                                int minOffset = affected_value - affectedBounds.upperBound;
                                int maxOffset = affected_value - affectedBounds.lowerBound;
                                if (System.Math.Sign(minOffset) != System.Math.Sign(maxOffset))
                                    throw new Exception($"Inconsistent offset between array indexer expressions: {mutated_index} and {affected_index}");
                                int offset = (minOffset > 0) ? minOffset : maxOffset;
                                if (offset != 0)
                                {
                                    IVariableDeclaration loopVar = GetOffsetVariable(mutated_index);
                                    offsets.Add(loopVar, offset, true);
                                }
                            }
                            else if (extraIndices != null)
                            {
                                IVariableDeclaration loopVar = GetOffsetVariable(mutated_index);
                                if (loopVar != null && !matchedIndices.Contains(loopVar))
                                    extraIndices.Add(loopVar);
                            }
                        }
                    }
                }
                else if (mutatesWithinOnly)
                    return false;  // expressions are incomparable
                else if (mutated_index is ILiteralExpression mutated_literal)
                {
                    int mutated_value = (int)mutated_literal.Value;
                    Bounds affectedBounds = GetBounds(affected_index, boundsInAffected);
                    if (affectedBounds != null)
                    {
                        if (!affectedBounds.Contains(mutated_value))
                            return false;
                    }
                    if (offsets != null)
                    {
                        // if affected=i-k and mutated=v (when i=j), then we write to v on iter j, and read on iter v+k, so offset is j-v-k
                        // offset = (loop counter of write) - (loop counter of read)
                        Bounds mutatedBounds = GetBounds(affected_index, boundsInMutated);
                        if (mutatedBounds != null)
                        {
                            // mutatedBounds = [jLower-k, jUpper-k]
                            int minOffset = mutatedBounds.lowerBound - mutated_value;
                            int maxOffset = mutatedBounds.upperBound - mutated_value;
                            if (System.Math.Sign(minOffset) != System.Math.Sign(maxOffset))
                                throw new Exception("Inconsistent offset between array indexer expressions: " + mutated_index + " and " + affected_index);
                            int offset = (minOffset > 0) ? minOffset : maxOffset;
                            if (offset != 0)
                            {
                                IVariableDeclaration affectedVar = GetOffsetVariable(affected_index);
                                offsets.Add(affectedVar, offset, true);
                            }
                        }
                    }
                }
                else if (offsets != null || extraIndices != null)
                {
                    // neither affected nor mutated is literal
                    // check for offsetting
                    IVariableDeclaration mutatedVar = GetOffsetVariable(mutated_index, out int mutated_offset);
                    IVariableDeclaration affectedVar = GetOffsetVariable(affected_index, out int affected_offset);
                    if (mutatedVar != null)
                    {
                        if (mutatedVar.Equals(affectedVar))
                        {
                            // mutated has the form [t + mutated_offset]
                            // affected has the form [t + affected_offset]
                            int offset = 0;
                            offset += affected_offset;
                            offset -= mutated_offset;
                            if (offset != 0)
                            {
                                bool isAvailable;
                                if (offset < 0)
                                {
                                    // TODO: make use of bounds
                                    // forward loop from 0 to size-1
                                    // we want to know if the first affected element will be mutated at an earlier time.
                                    // if affected_offset <= 0 then first affected is element 0 at time = -affected_offset.
                                    // if mutated_offset <= 0 then element 0 is mutated at time = -mutated_offset.
                                    // therefore available when -mutated_offset < -affected_offset, i.e. mutated_offset > affected_offset
                                    // which is always true.
                                    // if mutated_offset > 0 then element 0 is never mutated so not available.
                                    // if affected_offset > 0 then first affected is affected_offset at time 0 so not available.
                                    isAvailable = (affected_offset <= 0) && (mutated_offset <= 0);
                                }
                                else
                                {
                                    // backward loop from size-1 to 0
                                    // we want to know if the last affected element mutated will be mutated at a later time.
                                    // if affected_offset >= 0 then last affected is element size-1 at time = size-1 - affected_offset.
                                    // if mutated_offset >= 0 then this element is mutated at time = size-1 - mutated_offset.
                                    // therefore available when mutated_offset < affected_offset which is always true.
                                    // if mutated_offset < 0 then element size-1 is never mutated so not available.
                                    // if affected_offset < 0 then last affected is (size-1+affected_offset) at last time so not available.
                                    //isAvailable = (affected_offset >= 0); // TODO
                                    isAvailable = (affected_offset >= 0) && (mutated_offset >= 0);
                                }
                                offsets.Add(mutatedVar, offset, isAvailable);
                            }
                            matchedIndices.Add(mutatedVar);
                        }
                        else
                        {
                            // affected and mutated use two different loop variables.
                            // we cannot merge on either of these loops.
                            if(!matchedIndices.Contains(mutatedVar))
                                extraIndices.Add(mutatedVar);
                            if (affectedVar != null)
                                extraIndices.Add(affectedVar);
                        }
                    }
                    else
                    {
                        foreach(var v in GetVariables(mutated_index))
                        {
                            if(!matchedIndices.Contains(v))
                                extraIndices.Add(v);
                        }
                        if (affectedVar != null)
                            extraIndices.Add(affectedVar);
                    }
                }
            }
            return true;
        }

        /// <summary>
        /// Reflection cache
        /// </summary>
        readonly MethodInfo AnyMethod = new Func<object[], object>(FactorManager.Any).Method;

        /// <summary>
        /// Returns true if mutating the first expression would affect the value of the second.
        /// For example, if the first expression is 'x' and the second is 'x[1]' the answer is true.
        /// If the first expression is 'x[0]' and the second is 'x[1]' the answer is false.
        /// 
        /// If 'mutatesWithinOnly' is true, the result will only be true if the first expression mutates
        /// all or part of the second, but nothing else.  For example, if the first is 'x[0]' and the second is 'x'
        /// the result is true but not if the first is 'x' and the second 'x[0]'. 
        /// </summary>
        /// <param name="context">The context</param>
        /// <param name="mutated">The mutating expression</param>
        /// <param name="affected">The possibly affected expression</param>
        /// <param name="mutatesWithinOnly">See summary</param>
        /// <param name="mutatedStatement">Bindings on the index variables in mutated</param>
        /// <param name="affectedStatement">Bindings on the index variables in affected</param>
        /// <param name="boundsInMutated">Lower bounds on the index variables in mutated</param>
        /// <param name="boundsInAffected">Lower bounds on the index variables in affected</param>
        /// <param name="offsets">Modified to contain indexing offsets necessary for a match</param>
        /// <param name="extraIndices">Modified to contain loop indices in mutated that must be iterated to cover affected (i.e. loops that cannot be merged)</param>
        /// <returns></returns>
        internal bool MutatingFirstAffectsSecond(
            BasicTransformContext context,
            IExpression mutated,
            IExpression affected,
            bool mutatesWithinOnly,
            IStatement mutatedStatement = null,
            IStatement affectedStatement = null,
            IReadOnlyDictionary<IVariableDeclaration, Bounds> boundsInMutated = null,
            IReadOnlyDictionary<IVariableDeclaration, Bounds> boundsInAffected = null,
            OffsetInfo offsets = null,
            ICollection<IVariableDeclaration> extraIndices = null)
        {
            // if mutated has the special form Any(expr,expr,...) then check if all sub-expressions change affected
            if (IsStaticMethod(mutated, AnyMethod))
            {
                IMethodInvokeExpression imie = (IMethodInvokeExpression)mutated;
                foreach (IExpression arg in imie.Arguments)
                {
                    // TODO: what if offsets/extraIndices do not agree?
                    if (!MutatingFirstAffectsSecond(context, arg, affected, mutatesWithinOnly, mutatedStatement, affectedStatement, boundsInMutated, boundsInAffected, offsets, extraIndices))
                        return false;
                }
                return (imie.Arguments.Count > 0);
            }
            // if affected has the special form Any(expr,expr,...) then check if mutated changes any sub-expressions
            if (IsStaticMethod(affected, AnyMethod))
            {
                IMethodInvokeExpression imie = (IMethodInvokeExpression)affected;
                foreach (IExpression arg in imie.Arguments)
                {
                    // TODO: what if offsets/extraIndices do not agree?
                    if (MutatingFirstAffectsSecond(context, mutated, arg, mutatesWithinOnly, mutatedStatement, affectedStatement, boundsInMutated, boundsInAffected, offsets, extraIndices))
                        return true;
                }
                return false;
            }
            if(affectedStatement != null && mutatedStatement != null)
            {
                Set<IVariableDeclaration> localVarsInMutated = new Set<IVariableDeclaration>();
                var bindingsInMutated = GetBindings(mutatedStatement, localVarsInMutated);
                var bindingsInAffected = GetBindings(affectedStatement, null);
                if (!MutatingFirstAffectsSecond(mutated, affected, bindingsInMutated, bindingsInAffected, localVarsInMutated.Contains))
                    return false;
            }
            // Algorithm:
            // Examine the prefixes of the two expressions from left to right.
            // If the innermost targets don't match, return false.
            // If the prefixes have non-overlapping array indices, return false.
            // If the prefixes are incomparable, e.g. property versus field, return !mutatesWithinOnly (to be conservative)
            // If mutatesWithinOnly=true, return false unless we are sure.
            // If mutatesWithinOnly=false, return true unless we are sure.
            List<IExpression> mutatedPrefixes = GetAllPrefixes(mutated);
            List<IExpression> affectedPrefixes = GetAllPrefixes(affected);
            if (!mutatedPrefixes[0].Equals(affectedPrefixes[0]))
                return false;
            if (mutatesWithinOnly && affectedPrefixes.Count > mutatedPrefixes.Count)
                return false;
            int count = System.Math.Min(mutatedPrefixes.Count, affectedPrefixes.Count);
            Set<IVariableDeclaration> matchedIndices = new Set<IVariableDeclaration>();
            for (int i = 1; i < count; i++)
            {
                bool lastBracket = (i == affectedPrefixes.Count - 1);
                IExpression mutatedPrefix = mutatedPrefixes[i];
                IExpression affectedPrefix = affectedPrefixes[i];
                if (mutatedPrefix is IArrayIndexerExpression mutated_iaie)
                {
                    if (affectedPrefix is IArrayIndexerExpression affected_iaie)
                    {
                        try
                        {
                            if (!IndicesOverlap(mutated_iaie.Indices, affected_iaie.Indices, mutatesWithinOnly && lastBracket, boundsInMutated, boundsInAffected, offsets, extraIndices, matchedIndices))
                                return false;
                        }
                        catch (Exception ex)
                        {
                            throw new Exception($"Exception while comparing expressions '{mutated}' and '{affected}': {ex.Message}", ex);
                        }
                    }
                    else
                        return !mutatesWithinOnly;
                }
                else if (mutatedPrefix is IPropertyIndexerExpression mutated_ipie)
                {
                    if (affectedPrefix is IPropertyIndexerExpression affected_ipie)
                    {
                        if (!IndicesOverlap(mutated_ipie.Indices, affected_ipie.Indices, mutatesWithinOnly && lastBracket, boundsInMutated, boundsInAffected, offsets, extraIndices, matchedIndices))
                            return false;
                    }
                    else
                        return !mutatesWithinOnly;
                }
                else if (mutatedPrefix is IPropertyReferenceExpression mutated_ipre)
                {
                    if (affectedPrefix is IPropertyReferenceExpression affected_ipre)
                    {
                        if (!mutated_ipre.Property.Equals(affected_ipre.Property))
                            return !mutatesWithinOnly;
                    }
                    else
                        return !mutatesWithinOnly;
                }
                else if (mutatedPrefix is IFieldReferenceExpression mutated_ifre)
                {
                    if (affectedPrefix is IFieldReferenceExpression affected_ifre)
                    {
                        if (!mutated_ifre.Field.Equals(affected_ifre.Field))
                            return !mutatesWithinOnly;
                    }
                    else
                        return !mutatesWithinOnly;
                }
                else
                    throw new Exception("Unhandled expression type: " + mutatedPrefix);
            }
            if (extraIndices != null && mutatedPrefixes.Count > count)
            {
                // add extra indices in mutated to extraIndices
                // if mutated = array[i][j] and affected = array[i] then j is extra index
                // but if mutated = array[i][i] and affected = array[i] then i is not an extra index
                Containers containers = Containers.GetContainersNeededForExpression(context, affected);
                Set<IVariableDeclaration> loopVars = new Set<IVariableDeclaration>(ReferenceEqualityComparer<IVariableDeclaration>.Instance);
                foreach (IStatement container in containers.inputs)
                {
                    if (container is IForStatement ifs)
                    {
                        loopVars.Add(LoopVariable(ifs));
                    }
                }
                for (int i = count; i < mutatedPrefixes.Count; i++)
                {
                    IExpression mutatedPrefix = mutatedPrefixes[i];
                    if (mutatedPrefix is IArrayIndexerExpression mutated_iaie)
                    {
                        foreach (IExpression index in mutated_iaie.Indices)
                        {
                            foreach (var v in GetVariables(index))
                            {
                                if (!loopVars.Contains(v))
                                    extraIndices.Add(v);
                            }
                        }
                    }
                }
            }
            return true;
        }

        internal bool MutatingFirstAffectsSecond(
            IExpression mutated,
            IExpression affected,
            IReadOnlyCollection<ConditionBinding> bindingsInMutated,
            IReadOnlyCollection<ConditionBinding> bindingsInAffected,
            Func<IVariableDeclaration,bool> isLocalVarInMutated)
        {
            // if mutated has the special form Any(expr,expr,...) then check if all sub-expressions change affected
            if (IsStaticMethod(mutated, AnyMethod))
            {
                IMethodInvokeExpression imie = (IMethodInvokeExpression)mutated;
                foreach (IExpression arg in imie.Arguments)
                {
                    // TODO: what if offsets/extraIndices do not agree?
                    if (!MutatingFirstAffectsSecond(arg, affected, bindingsInMutated, bindingsInAffected, isLocalVarInMutated))
                        return false;
                }
                return (imie.Arguments.Count > 0);
            }
            // if affected has the special form Any(expr,expr,...) then check if mutated changes any sub-expressions
            if (IsStaticMethod(affected, AnyMethod))
            {
                IMethodInvokeExpression imie = (IMethodInvokeExpression)affected;
                foreach (IExpression arg in imie.Arguments)
                {
                    // TODO: what if offsets/extraIndices do not agree?
                    if (MutatingFirstAffectsSecond(mutated, arg, bindingsInMutated, bindingsInAffected, isLocalVarInMutated))
                        return true;
                }
                return false;
            }

            return !AreDisjoint(mutated, bindingsInMutated, affected, bindingsInAffected, isLocalVarInMutated);
        }

        /// <summary>
        /// Augment the bindings with the minimal conditions necessary for the two expressions to overlap
        /// </summary>
        /// <param name="e1"></param>
        /// <param name="e2"></param>
        /// <param name="bindings1">Bindings for e1</param>
        /// <param name="bindings2">Bindings for e2</param>
        /// <returns>True if overlap conditions could be found, false if the expressions cannot overlap</returns>
        internal bool AddIndexBindings(IExpression e1, IExpression e2, ICollection<ConditionBinding> bindings1, ICollection<ConditionBinding> bindings2)
        {
            int tempCount = 0;
            List<IExpression> prefixes1 = GetAllPrefixes(e1);
            List<IExpression> prefixes2 = GetAllPrefixes(e2);
            if (!prefixes1[0].Equals(prefixes2[0]))
                return false;
            int count = System.Math.Min(prefixes1.Count, prefixes2.Count);
            for (int i = 1; i < count; i++)
            {
                IExpression prefix1 = prefixes1[i];
                IExpression prefix2 = prefixes2[i];
                if (prefix1 is IArrayIndexerExpression iaie1)
                {
                    if (prefix2 is IArrayIndexerExpression iaie2)
                    {
                        if (iaie1.Indices.Count != iaie2.Indices.Count)
                            break;
                        for (int j = 0; j < iaie1.Indices.Count; j++)
                        {
                            IExpression index1 = iaie1.Indices[j];
                            IExpression index2 = iaie2.Indices[j];
                            if (index1 is ILiteralExpression && index2 is ILiteralExpression)
                            {
                                if (!index1.Equals(index2))
                                    return false;
                            }
                            else
                            {
                                // add a condition for the indices to be equal
                                AddBinding(index1, index2, bindings1, bindings2, tempCount++);
                            }
                        }
                    }
                    else
                        break;
                }
                else if (prefix1 is IPropertyIndexerExpression ipie1)
                {
                    if (prefix2 is IPropertyIndexerExpression ipie2)
                    {
                        if (ipie1.Indices.Count != ipie2.Indices.Count)
                            break;
                        for (int j = 0; j < ipie1.Indices.Count; j++)
                        {
                            IExpression index1 = ipie1.Indices[j];
                            IExpression index2 = ipie2.Indices[j];
                            if (index1 is ILiteralExpression && index2 is ILiteralExpression)
                            {
                                if (!index1.Equals(index2))
                                    return false;
                            }
                            else
                            {
                                // add a condition for the indices to be equal
                                AddBinding(index1, index2, bindings1, bindings2, tempCount++);
                            }
                        }
                    }
                    else
                        break;
                }
                else if (prefix1 is IPropertyReferenceExpression ipre1)
                {
                    if (prefix2 is IPropertyReferenceExpression ipre2)
                    {
                        // we assume that mutating one property does not affect another.
                        if (!ipre1.Property.Equals(ipre2.Property))
                            break;
                    }
                    else
                        break;
                }
                else if (prefix1 is IFieldReferenceExpression ifre1)
                {
                    if (prefix2 is IFieldReferenceExpression ifre2)
                    {
                        if (!ifre1.Field.Equals(ifre2.Field))
                            break;
                    }
                    else
                        break;
                }
                else if (!prefix1.Equals(prefix2))
                {
                    throw new Exception("Unhandled expression type: " + prefix1);
                }
            }
            return true;
        }

        /// <summary>
        /// Augment both bindings to ensure that (index1 == index2)
        /// </summary>
        /// <param name="index1"></param>
        /// <param name="index2"></param>
        /// <param name="bindings1"></param>
        /// <param name="bindings2"></param>
        /// <param name="tempCount">A counter for numbering temporary variables</param>
        private static void AddBinding(IExpression index1, IExpression index2, ICollection<ConditionBinding> bindings1, ICollection<ConditionBinding> bindings2,
                                       int tempCount)
        {
            if (index1 is ILiteralExpression)
            {
                bindings2.Add(new ConditionBinding(index2, index1));
            }
            else if (index2 is ILiteralExpression)
            {
                bindings1.Add(new ConditionBinding(index1, index2));
            }
            else
            {
                Type exprType = index1.GetExpressionType();
                IVariableDeclaration tempVar = Builder.VarDecl("_t" + tempCount, exprType);
                IExpression tempRef = Builder.VarRefExpr(tempVar);
                bindings1.Add(new ConditionBinding(index1, tempRef));
                bindings2.Add(new ConditionBinding(tempRef, index2));
            }
        }

        /// <summary>
        /// Returns true if the expressions access distinct array elements or have disjoint condition contexts
        /// </summary>
        /// <param name="expr1"></param>
        /// <param name="ebBinding"></param>
        /// <param name="expr2"></param>
        /// <param name="eb2Binding"></param>
        /// <param name="isLoopVar"></param>
        /// <returns></returns>
        internal bool AreDisjoint(
            IExpression expr1,
            IReadOnlyCollection<ConditionBinding> ebBinding,
            IExpression expr2,
            IReadOnlyCollection<ConditionBinding> eb2Binding,
            Func<IVariableDeclaration,bool> isLoopVar)
        {
            var bindings1 = new Set<ConditionBinding>();
            bindings1.AddRange(ebBinding);
            var bindings2 = new Set<ConditionBinding>();
            bindings2.AddRange(eb2Binding);
            // augment the two bindings to make the expressions overlap
            bool match = AddIndexBindings(expr1, expr2, bindings1, bindings2);
            if (!match)
                return true;
            // remove loop vars from bindings1 (see InferTests.ConstrainBetweenTest3)
            var bindings1Reduced = RemoveLoopVars(bindings1, isLoopVar);
            if (bindings1Reduced == null)
                return true;
            bindings2.AddRange(bindings1Reduced);
            // bindings2 is now the combined set of conditions.  If we find a contradiction, the uses are disjoint.
            var bindings2Reduced = FindContradiction(bindings2, expr => !(expr is ILiteralExpression));
            return (bindings2Reduced == null);
        }

        internal IReadOnlyCollection<ConditionBinding> RemoveLoopVars(IReadOnlyCollection<ConditionBinding> bindings, Func<IVariableDeclaration,bool> isLoopVar)
        {
            return FindContradiction(bindings, expr => ContainsLoopVars(expr, isLoopVar));
        }

        /// <summary>
        /// Find a contradiction in a set of conditions, i.e. prove there is no satisfying assignment
        /// </summary>
        /// <param name="bindings">A set of conditions.  Not modified.</param>
        /// <param name="predicate">Indicates the bindings to eliminate.  If null, all bindings are eliminated.</param>
        /// <returns>null if inconsistent, otherwise a reduced list of bindings where none satisfy the predicate</returns>
        internal IReadOnlyCollection<ConditionBinding> FindContradiction(IReadOnlyCollection<ConditionBinding> bindings, Predicate<IExpression> predicate = null)
        {
            while (bindings.Count > 0)
            {
                // find the binding with minimum depth
                ConditionBinding currentBinding = null;
                IExpression exprFind = null;
                IExpression exprReplace = null;
                int minDepth = int.MaxValue;
                foreach (ConditionBinding binding in bindings)
                {
                    if (!IsValid(binding))
                        return null;
                    bool containsLhs = (predicate == null) || predicate(binding.lhs);
                    if (containsLhs)
                    {
                        int depthLhs = GetExpressionDepth(binding.lhs);
                        if (depthLhs < minDepth)
                        {
                            minDepth = depthLhs;
                            currentBinding = binding;
                            exprFind = binding.lhs;
                            exprReplace = binding.rhs;
                        }
                    }
                    bool containsRhs = (predicate == null) || predicate(binding.rhs);
                    if (containsRhs)
                    {
                        int depthRhs = GetExpressionDepth(binding.rhs);
                        if (depthRhs < minDepth)
                        {
                            minDepth = depthRhs;
                            currentBinding = binding;
                            exprFind = binding.rhs;
                            exprReplace = binding.lhs;
                        }
                    }
                }
                if (currentBinding == null)
                    return bindings;
                // apply the binding
                var newBindings = new Set<ConditionBinding>();
                foreach (ConditionBinding binding in bindings)
                {
                    if (object.ReferenceEquals(binding, currentBinding))
                        continue;
                    // could skip cases where we know exprFind will not appear, e.g. exprs not satisfying the predicate
                    // or whose depth is too small.
                    IExpression newLhs = Builder.ReplaceExpression(binding.lhs, exprFind, exprReplace);
                    IExpression newRhs = Builder.ReplaceExpression(binding.rhs, exprFind, exprReplace);
                    newBindings.Add(new ConditionBinding(newLhs, newRhs));
                }
                bindings = newBindings;
            }
            return bindings;
        }

        private static int GetExpressionDepth(IExpression expr)
        {
            if (expr == null)
                return 0;
            else if (expr is IArrayIndexerExpression iaie)
            {
                int maxDepth = GetExpressionDepth(iaie.Target);
                foreach (IExpression index in iaie.Indices)
                {
                    int depth = GetExpressionDepth(index);
                    if (depth > maxDepth)
                        maxDepth = depth;
                }
                return 1 + maxDepth;
            }
            else if (expr is IMethodReferenceExpression imre)
            {
                return GetExpressionDepth(imre.Target);
            }
            else if (expr is IMethodInvokeExpression imie)
            {
                int maxDepth = GetExpressionDepth(imie.Method);
                foreach (IExpression arg in imie.Arguments)
                {
                    int depth = GetExpressionDepth(arg);
                    if (depth > maxDepth)
                        maxDepth = depth;
                }
                return 1 + maxDepth;
            }
            else if (expr is IBinaryExpression ibe)
            {
                return 1 + System.Math.Max(GetExpressionDepth(ibe.Left), GetExpressionDepth(ibe.Right));
            }
            else if (expr is IUnaryExpression iue)
            {
                return 1 + GetExpressionDepth(iue.Expression);
            }
            else if (expr is IPropertyReferenceExpression ipre)
            {
                return 1 + GetExpressionDepth(ipre.Target);
            }
            else if (expr is IPropertyIndexerExpression ipie)
            {
                int maxDepth = GetExpressionDepth(ipie.Target);
                foreach (IExpression index in ipie.Indices)
                {
                    int depth = GetExpressionDepth(index);
                    if (depth > maxDepth)
                        maxDepth = depth;
                }
                return 1 + maxDepth;
            }
            else if (expr is IObjectCreateExpression ioce)
            {
                int maxDepth = GetExpressionDepth(ioce.Initializer);
                foreach (IExpression arg in ioce.Arguments)
                {
                    int depth = GetExpressionDepth(arg);
                    if (depth > maxDepth)
                        maxDepth = depth;
                }
                return 1 + maxDepth;
            }
            else if (expr is IAssignExpression iae)
            {
                return 1 + System.Math.Max(GetExpressionDepth(iae.Target), GetExpressionDepth(iae.Expression));
            }
            else if (expr is IVariableReferenceExpression || expr is ILiteralExpression || expr is IArgumentReferenceExpression || expr is ITypeReferenceExpression
                || expr is IVariableDeclarationExpression || expr is IArrayCreateExpression)
            {
                return 1;
            }
            else if (expr is IAddressOutExpression iaoe)
            {
                return 2;
            }
            else
                throw new NotImplementedException();
        }

        private bool ContainsLoopVars(IExpression expr, Func<IVariableDeclaration,bool> isLoopVar)
        {
            return GetVariables(expr).Any(isLoopVar);
        }

        private bool IsValid(ConditionBinding binding)
        {
            if (TryEvaluate<object>(binding.GetExpression(), null, out object value))
                return (bool)value;
            else
                return true;
        }

        private IEnumerable<IExpression> GetSummands(IExpression expr)
        {
            if (expr is IBinaryExpression ibe)
            {
                if (ibe.Operator == BinaryOperator.Add)
                {
                    foreach (var summand in GetSummands(ibe.Left))
                        yield return summand;
                    foreach (var summand in GetSummands(ibe.Right))
                        yield return summand;
                }
                else if (ibe.Operator == BinaryOperator.Subtract)
                {
                    foreach (var summand in GetSummands(ibe.Left))
                        yield return summand;
                    foreach (var summand in GetSummands(ibe.Right))
                        yield return Negate(summand);
                }
                else
                    yield return expr;
            }
            else if (expr is IMethodInvokeExpression imie)
            {
                if (IsStaticMethod(imie, new Func<int, int, int>(ML.Probabilistic.Factors.Factor.Plus)))
                {
                    foreach (var arg in imie.Arguments)
                    {
                        foreach (var summand in GetSummands(arg))
                            yield return summand;
                    }
                }
                else if (IsStaticMethod(imie, new Func<int, int, int>(ML.Probabilistic.Factors.Factor.Difference)))
                {
                    foreach (var summand in GetSummands(imie.Arguments[0]))
                        yield return summand;
                    foreach (var summand in GetSummands(imie.Arguments[1]))
                        yield return Negate(summand);
                }
                else
                    yield return expr;
            }
            else
                yield return expr;
        }

        private IExpression Negate(IExpression expr)
        {
            if (expr is IMethodInvokeExpression imie)
            {
                if (IsStaticMethod(imie, new Func<int, int>(ML.Probabilistic.Factors.Factor.Negate)))
                {
                    return imie.Arguments[0];
                }
            }
            if (expr is IUnaryExpression iue)
            {
                if (iue.Operator == UnaryOperator.Negate)
                {
                    return iue.Expression;
                }
            }
            bool useOperator = true;
            if (useOperator)
                return Builder.UnaryExpr(UnaryOperator.Negate, expr);
            else
                return Builder.StaticMethod(new Func<int, int>(ML.Probabilistic.Factors.Factor.Negate), expr);
        }

        public bool TryEvaluate<T>(IExpression expr, IDictionary<IVariableDeclaration, T> bindings, out T value)
        {
            if (expr is IVariableReferenceExpression && bindings != null)
            {
                IVariableDeclaration ivd = GetVariableDeclaration(expr);
                return bindings.TryGetValue(ivd, out value);
            }
            else if (expr is IBinaryExpression ibe)
            {
                if (ibe.Operator == BinaryOperator.ValueInequality ||
                    ibe.Operator == BinaryOperator.IdentityInequality ||
                    ibe.Operator == BinaryOperator.GreaterThan ||
                    ibe.Operator == BinaryOperator.LessThan)
                {
                    if (ibe.Left.Equals(ibe.Right))
                    {
                        value = (T)(object)false;
                        return true;
                    }
                }
                else if (ibe.Operator == BinaryOperator.IdentityEquality || ibe.Operator == BinaryOperator.ValueEquality)
                {
                    if (ibe.Left.Equals(ibe.Right))
                    {
                        value = (T)(object)true;
                        return true;
                    }
                    if (ibe.Left.GetExpressionType().Equals(typeof(int)))
                    {
                        List<IExpression> summandsLeft = GetSummands(ibe.Left).ToList();
                        List<IExpression> summandsRight = new List<IExpression>();
                        bool foundMatch = false;
                        // Cancel identical expressions on both sides of the equality.
                        foreach (var summand in GetSummands(ibe.Right))
                        {
                            if (summandsLeft.Contains(summand))
                            {
                                foundMatch = true;
                                summandsLeft.Remove(summand);
                            }
                            else
                                summandsRight.Add(summand);
                        }
                        if (foundMatch)
                        {
                            if (summandsLeft.Count == 0)
                                summandsLeft.Add(Builder.LiteralExpr(0));
                            if (summandsRight.Count == 0)
                                summandsRight.Add(Builder.LiteralExpr(0));
                            if (summandsLeft.Count == 1 && summandsRight.Count == 1)
                            {
                                ibe = Builder.BinaryExpr(summandsLeft[0], ibe.Operator, summandsRight[0]);
                                // fall through
                            }
                        }
                    }
                }
                if (TryEvaluate(ibe.Left, bindings, out T left) && TryEvaluate(ibe.Right, bindings, out T right))
                {
                    // must use runtime type here, not T
                    Type type = left.GetType();
                    value = (T)Microsoft.ML.Probabilistic.Compiler.Reflection.Invoker.InvokeStatic(type, ExpressionEvaluator.binaryOperatorNames[(int)ibe.Operator], left, right);
                    return true;
                }
            }
            else if (expr is IUnaryExpression iue)
            {
                if (TryEvaluate(iue.Expression, bindings, out T target))
                {
                    // must use runtime type here, not T
                    Type type = target.GetType();
                    value = (T)Microsoft.ML.Probabilistic.Compiler.Reflection.Invoker.InvokeStatic(type, ExpressionEvaluator.unaryOperatorNames[(int)iue.Operator], target);
                    return true;
                }
            }
            else if (expr is ILiteralExpression ile)
            {
                if (ile.Value is T)
                {
                    value = (T)ile.Value;
                    return true;
                }
            }
            value = default(T);
            return false;
        }

        /// <summary>
        /// Returns true if expr is on the left hand side of the innermost assignment statement in the context stack.
        /// </summary>
        /// <param name="context"></param>
        /// <param name="expr"></param>
        /// <returns></returns>
        public bool IsOnLHSOfAssignment(BasicTransformContext context, IExpression expr)
        {
            int assignIndex = context.FindAncestorNotSelfIndex<IAssignExpression>();
            if (assignIndex == -1)
                return false;
            IAssignExpression iae = (IAssignExpression)context.GetAncestor(assignIndex);
            if (iae.Target == context.GetAncestor(assignIndex + 1))
                return IsPartOf(iae.Target, expr);
            return false;
        }

        public int GetAncestorIndexOfLoopBeingInitialized(BasicTransformContext context)
        {
            int forIndex = context.FindAncestorIndex<IForStatement>();
            if (forIndex == -1)
                return -1;
            IForStatement forStmt = (IForStatement)context.GetAncestor(forIndex);
            if (context.InputStack.Count > forIndex + 1 && context.InputStack[forIndex + 1].inputElement.Equals(forStmt.Initializer))
            {
                return forIndex;
            }
            else
                return -1;
        }

        public bool IsBeingAllocated(BasicTransformContext context, IExpression expr)
        {
            if (IsOnLHSOfAssignment(context, expr))
            {
                IAssignExpression iae = context.FindAncestor<IAssignExpression>();
                if (iae.Expression is IArrayCreateExpression)
                    return true;
            }
            return false;
        }

        public bool IsTypeReferenceTo(ITypeReferenceExpression itre, Type type)
        {
            return itre.Type.DotNetType == type;
        }

        public bool IsLiteral(IExpression expr, object val)
        {
            if (expr is ILiteralExpression ile)
            {
                return ile.Value.Equals(val);
            }
            return false;
        }

        public T GetLiteral<T>(IExpression expr)
        {
            if (expr is ILiteralExpression ile)
            {
                return (T)ile.Value;
            }
            return default(T);
        }

        public IVariableDeclaration LoopVariable(IForStatement ifs)
        {
            IStatement ist = ifs.Initializer;
            if (ist is IBlockStatement ibs)
            {
                if (ibs.Statements.Count != 1)
                    throw new NotSupportedException("For statement has multi-statement initializer:" + ifs);
                ist = ibs.Statements[0];
            }
            IExpressionStatement init = (IExpressionStatement)ist;
            IAssignExpression iae = (IAssignExpression)init.Expression;
            IVariableDeclaration ivd = GetVariableDeclaration(iae.Target);
            return ivd;
        }

        public IExpression LoopSizeExpression(IForStatement loop)
        {
            if (loop.Condition is IBinaryExpression ibe)
            {
                if (ibe.Operator == BinaryOperator.LessThan)
                {
                    return ibe.Right;
                }
                else if (ibe.Operator == BinaryOperator.GreaterThanOrEqual)
                {
                    var start = LoopStartExpression(loop);
                    if (start is IBinaryExpression ibe2)
                    {
                        if (ibe2.Operator == BinaryOperator.Subtract)
                        {
                            return ibe2.Left;
                        }
                    }
                }
                throw new ArgumentException("Unrecognized loop syntax");
            }
            else
            {
                throw new ArgumentException("Loop condition is not a BinaryExpression");
            }
        }

        public IStatement LoopBreakStatement(IForStatement loop)
        {
            return Builder.AssignStmt(Builder.VarRefExpr(LoopVariable(loop)), LoopBreakExpression(loop));
        }

        private IExpression LoopBreakExpression(IForStatement loop)
        {
            if (loop.Condition is IBinaryExpression ibe)
            {
                if (ibe.Operator == BinaryOperator.LessThan)
                    return Builder.BinaryExpr(BinaryOperator.Subtract, ibe.Right, Builder.LiteralExpr(1));
                else if (ibe.Operator == BinaryOperator.LessThanOrEqual)
                    return ibe.Right;
                else if (ibe.Operator == BinaryOperator.GreaterThanOrEqual)
                    return ibe.Right;
                else if (ibe.Operator == BinaryOperator.GreaterThan)
                    return Builder.BinaryExpr(BinaryOperator.Add, ibe.Right, Builder.LiteralExpr(1));
                else
                    throw new ArgumentException($"Unrecognized loop condition: {ibe}");
            }
            else
            {
                throw new ArgumentException("Loop condition is not a BinaryExpression");
            }
        }

        public IExpression LoopStartExpression(IForStatement loop)
        {
            if (loop.Initializer is IExpressionStatement ies)
            {
                if (ies.Expression is IAssignExpression iae)
                {
                    return iae.Expression;
                }
            }
            throw new ArgumentException("Loop initializer is not an assignment");
        }

        public class Bounds
        {
            public int lowerBound = int.MinValue, upperBound = int.MaxValue;

            public bool Contains(int value)
            {
                return (value >= lowerBound) && (value <= upperBound);
            }

            public override string ToString()
            {
                return String.Format("[{0},{1}]", lowerBound, upperBound);
            }
        }

        public void AddLoopBounds(Dictionary<IVariableDeclaration, Bounds> bounds, IStatement ist)
        {
            if (ist is IForStatement ifs)
            {
                IVariableDeclaration loopVar = LoopVariable(ifs);
                Bounds b;
                if (!bounds.TryGetValue(loopVar, out b))
                {
                    b = new Bounds();
                    bounds[loopVar] = b;
                }
                IExpression start = LoopStartExpression(ifs);
                if (start is ILiteralExpression startLiteral)
                {
                    int startValue = (int)startLiteral.Value;
                    b.lowerBound = System.Math.Max(b.lowerBound, startValue);
                }
                IExpression size = LoopSizeExpression(ifs);
                if (size is ILiteralExpression sizeLiteral)
                {
                    int endValue = (int)sizeLiteral.Value - 1;
                    b.upperBound = System.Math.Min(b.upperBound, endValue);
                }
                if (ifs.Body.Statements.Count == 1)
                    AddLoopBounds(bounds, ifs.Body.Statements[0]);
            }
            else if (ist is IConditionStatement ics)
            {
                IExpression condition = ics.Condition;
                if (condition is IBinaryExpression ibe)
                {
                    if (ibe.Left is IVariableReferenceExpression)
                    {
                        IVariableDeclaration loopVar = GetVariableDeclaration(ibe.Left);
                        Bounds b;
                        if (!bounds.TryGetValue(loopVar, out b))
                        {
                            b = new Bounds();
                            bounds[loopVar] = b;
                        }
                        if (ibe.Left.GetExpressionType().Equals(typeof(int)) && ibe.Right is ILiteralExpression right)
                        {
                            int value = (int)right.Value;
                            if (ibe.Operator == BinaryOperator.GreaterThan)
                            {
                                b.lowerBound = System.Math.Max(b.lowerBound, value + 1);
                            }
                            else if (ibe.Operator == BinaryOperator.GreaterThanOrEqual)
                            {
                                b.lowerBound = System.Math.Max(b.lowerBound, value);
                            }
                            else if (ibe.Operator == BinaryOperator.LessThanOrEqual)
                            {
                                b.upperBound = System.Math.Min(b.upperBound, value);
                            }
                            else if (ibe.Operator == BinaryOperator.LessThan)
                            {
                                b.upperBound = System.Math.Min(b.upperBound, value - 1);
                            }
                            else if (ibe.Operator == BinaryOperator.ValueEquality)
                            {
                                b.lowerBound = System.Math.Max(b.lowerBound, value);
                                b.upperBound = System.Math.Min(b.upperBound, value);
                            }
                            else if (ibe.Operator == BinaryOperator.ValueInequality)
                            {
                                if (b.lowerBound == value)
                                    b.lowerBound++;
                                if (b.upperBound == value)
                                    b.upperBound--;
                            }
                        }
                    }
                    if (ibe.Right is IVariableReferenceExpression)
                    {
                        IVariableDeclaration loopVar = GetVariableDeclaration(ibe.Right);
                        Bounds b;
                        if (!bounds.TryGetValue(loopVar, out b))
                        {
                            b = new Bounds();
                            bounds[loopVar] = b;
                        }
                        if (ibe.Right.GetExpressionType().Equals(typeof(int)) && ibe.Left is ILiteralExpression left)
                        {
                            int value = (int)left.Value;
                            if (ibe.Operator == BinaryOperator.GreaterThan)
                            {
                                b.upperBound = System.Math.Min(b.upperBound, value - 1);
                            }
                            else if (ibe.Operator == BinaryOperator.GreaterThanOrEqual)
                            {
                                b.upperBound = System.Math.Min(b.upperBound, value);
                            }
                            else if (ibe.Operator == BinaryOperator.LessThanOrEqual)
                            {
                                b.lowerBound = System.Math.Max(b.lowerBound, value);
                            }
                            else if (ibe.Operator == BinaryOperator.LessThan)
                            {
                                b.lowerBound = System.Math.Max(b.lowerBound, value + 1);
                            }
                            else if (ibe.Operator == BinaryOperator.ValueEquality)
                            {
                                b.lowerBound = System.Math.Max(b.lowerBound, value);
                                b.upperBound = System.Math.Min(b.upperBound, value);
                            }
                            else if (ibe.Operator == BinaryOperator.ValueInequality)
                            {
                                if (b.lowerBound == value)
                                    b.lowerBound++;
                                if (b.upperBound == value)
                                    b.upperBound--;
                            }
                        }
                    }
                }
                if (ics.Then.Statements.Count == 1)
                    AddLoopBounds(bounds, ics.Then.Statements[0]);
            }
        }

        internal Set<ConditionBinding> GetBindings(IStatement stmt)
        {
            var bounds = new Dictionary<IVariableDeclaration, Bounds>(ReferenceEqualityComparer<IVariableDeclaration>.Instance);
            AddLoopBounds(bounds, stmt);
            var bindings = new Set<ConditionBinding>();
            foreach (var entry in bounds)
            {
                var b = entry.Value;
                if (b.lowerBound == b.upperBound)
                {
                    var binding = new ConditionBinding(Builder.VarRefExpr(entry.Key), Builder.LiteralExpr(b.lowerBound));
                    bindings.Add(binding);
                }
            }
            return bindings;
        }

        internal Set<ConditionBinding> GetBindings(IStatement ist, Set<IVariableDeclaration> localVars)
        {
            Set<ConditionBinding> bindings = new Set<ConditionBinding>();
            while (true)
            {
                if (ist is IForStatement ifs)
                {
                    IVariableDeclaration loopVar = LoopVariable(ifs);
                    if (localVars != null)
                        localVars.Add(loopVar);
                    IExpression start = LoopStartExpression(ifs);
                    if (IsForwardLoop(ifs))
                    {
                        bindings.Add(new ConditionBinding(Builder.BinaryExpr(Builder.VarRefExpr(loopVar), BinaryOperator.GreaterThanOrEqual, start)));
                    }
                    bindings.Add(new ConditionBinding(ifs.Condition));
                    if (ifs.Body.Statements.Count == 1)
                        ist = ifs.Body.Statements[0];
                    else
                        break;
                }
                else if (ist is IConditionStatement ics)
                {
                    bindings.Add(new ConditionBinding(ics.Condition));
                    if (ics.Then.Statements.Count == 1)
                        ist = ics.Then.Statements[0];
                    else
                        break;
                }
                else
                    break;
            }
            return bindings;
        }

        /// <summary>
        /// Returns true if any of the reference expressions refer to the specified index variable.
        /// </summary>
        /// <param name="refs"></param>
        /// <param name="indexVar"></param>
        /// <returns></returns>
        public bool IsIndexedBy(List<IVariableReferenceExpression> refs, IVariableDeclaration indexVar)
        {
            foreach (IArrayIndexerExpression iaie in refs)
            {
                foreach (IExpression indExpr in iaie.Indices)
                {
                    IVariableDeclaration ivd2 = ((IVariableReferenceExpression)indExpr).Variable.Resolve();
                    if (indexVar.Equals(ivd2))
                        return true;
                }
            }
            return false;
        }

        public IExpression StripIndexers(IExpression expr)
        {
            return StripIndexers(expr, false);
        }

        public IExpression StripIndexers(IExpression expr, bool varsOnly)
        {
            if (!(expr is IArrayIndexerExpression iaie))
                return expr;
            if ((!(iaie.Indices[0] is IVariableReferenceExpression)) && varsOnly)
                return expr;
            return StripIndexers(iaie.Target, varsOnly);
        }

        public IExpression StripFieldsAndProperties(IExpression expr)
        {
            if (expr is IPropertyReferenceExpression ipre)
                return ipre.Target;
            else if (expr is IFieldReferenceExpression ifre)
                return ifre.Target;
            else
                return expr;
        }

        public bool IsNewObject(IExpression expr, Type type)
        {
            if (!(expr is IObjectCreateExpression ioce))
                return false;
            Type t = Builder.ToType(ioce.Type);
            return t.Equals(type);
        }

        /// <summary>
        /// Returns true if the first expression is equal to, or a subarray or element of the second expression.
        /// For example, x, x[0], x[0][0] are all subarrays of x, but x[1] is not part of x[0].
        /// </summary>
        /// <param name="expr"></param>
        /// <param name="arrayExpr"></param>
        /// <returns></returns>
        public bool IsPartOf(IExpression expr, IExpression arrayExpr)
        {
            while (true)
            {
                if (arrayExpr.Equals(expr))
                    return true;
                if (expr is IArrayIndexerExpression iaie)
                    expr = iaie.Target;
                else
                {
                    IExpression oldexpr = expr;
                    expr = StripFieldsAndProperties(expr);
                    if (object.ReferenceEquals(oldexpr, expr))
                        return false;
                }
            }
        }

        /// <summary>
        /// Extracts the declaration of the parameter reference from a, possibly indexed, argument reference expression.
        /// Returns null if the expression is not either of these.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public IParameterDeclaration GetParameterDeclaration(IExpression expr)
        {
            expr = GetTarget(expr);
            if (expr is IArgumentReferenceExpression iare)
                return iare.Parameter.Resolve();
            else
                return null;
        }

        /// <summary>
        /// Extracts the field reference from a, possibly indexed, field reference expression.
        /// Returns null if the expression is not either of these.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public IFieldReference GetFieldReference(IExpression expr)
        {
            expr = GetTarget(expr);
            if (expr is IFieldReferenceExpression ifre)
                return ifre.Field;
            else
                return null;
        }

        /// <summary>
        /// Extracts the variable declaration from a, possibly indexed, reference or declaration expression.
        /// Returns null if the expression is not either of these.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public IVariableDeclaration GetVariableDeclaration(IExpression expr)
        {
            expr = GetTarget(expr);
            if (expr is IVariableReferenceExpression ivre)
                return ivre.Variable.Resolve();
            else if (expr is IVariableDeclarationExpression ivde)
                return ivde.Variable;
            else
                return null;
        }

        /// <summary>
        /// Extracts the variable or parameter declaration from a, possibly indexed, reference or declaration expression.
        /// Returns null if the expression is not either of these.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public object GetDeclaration(IExpression expr)
        {
            object decl = GetVariableDeclaration(expr);
            if (decl != null)
                return decl;
            decl = GetParameterDeclaration(expr);
            if (decl != null)
                return decl;
            return GetFieldReference(expr);
        }

        public object GetArrayDeclaration(IExpression expr)
        {
            expr = StripIndexers(expr);
            if (expr is IVariableReferenceExpression ivre)
                return ivre.Variable.Resolve();
            else if (expr is IVariableDeclarationExpression ivde)
                return ivde.Variable;
            else if (expr is IArgumentReferenceExpression iare)
                return iare.Parameter.Resolve();
            else if (expr is IFieldReferenceExpression ifre)
                return ifre.Field.Resolve();
            else if (expr is IPropertyReferenceExpression ipre)
                return ipre.Property.Resolve();
            else
                return null;
        }

        /// <summary>
        /// Get the innermost target of an expression, e.g. x[0].field[1].method(y) returns 'x'
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public IExpression GetTarget(IExpression expr)
        {
            IExpression target = null;
            ForEachPrefix(expr, prefix =>
            {
                if (target == null) target = prefix;
            });
            return target;
        }

        /// <summary>
        /// Get a list of prefixes of the given expression, starting from the innermost target, up to and including the given expression.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public List<IExpression> GetAllPrefixes(IExpression expr)
        {
            List<IExpression> result = new List<IExpression>();
            ForEachPrefix(expr, result.Add);
            return result;
        }

        private void ForEachPrefix(IExpression expr, Action<IExpression> action)
        {
            // This method must be kept consistent with GetTargets.
            if (expr is IArrayIndexerExpression iaie)
                ForEachPrefix(iaie.Target, action);
            else if (expr is IAddressOutExpression iaoe)
                ForEachPrefix(iaoe.Expression, action);
            else if (expr is IPropertyReferenceExpression ipre)
                ForEachPrefix(ipre.Target, action);
            else if (expr is IFieldReferenceExpression ifre)
            {
                IExpression target = ifre.Target;
                if (!(target is IThisReferenceExpression))
                    ForEachPrefix(target, action);
            }
            else if (expr is ICastExpression ice)
                ForEachPrefix(ice.Expression, action);
            else if (expr is IPropertyIndexerExpression ipie)
                ForEachPrefix(ipie.Target, action);
            else if (expr is IEventReferenceExpression iere)
                ForEachPrefix(iere.Target, action);
            else if (expr is IUnaryExpression iue)
                ForEachPrefix(iue.Expression, action);
            else if (expr is IAddressReferenceExpression iare)
                ForEachPrefix(iare.Expression, action);
            else if (expr is IMethodInvokeExpression imie)
                ForEachPrefix(imie.Method, action);
            else if (expr is IMethodReferenceExpression imre)
                ForEachPrefix(imre.Target, action);
            else if (expr is IDelegateInvokeExpression idie)
                ForEachPrefix(idie.Target, action);
            action(expr);
        }

        /// <summary>
        /// Get all variable references in an expression.  The same variable may appear more than once.
        /// </summary>
        /// <param name="expr">Any expression</param>
        /// <returns></returns>
        public IEnumerable<IVariableDeclaration> GetVariables(IExpression expr)
        {
            foreach(var decl in GetVariablesAndParameters(expr))
            {
                if (decl is IVariableDeclaration ivd)
                    yield return ivd;
            }
        }

        /// <summary>
        /// Get all variable and parameter references in an expression.  The same variable may appear more than once.
        /// </summary>
        /// <param name="expr">Any expression</param>
        /// <returns></returns>
        public IEnumerable<object> GetVariablesAndParameters(IExpression expr)
        {
            if (expr is IArrayIndexerExpression iaie)
            {
                foreach (IExpression index in iaie.Indices)
                {
                    foreach (var decl in GetVariablesAndParameters(index))
                    {
                        yield return decl;
                    }
                }
                foreach (var decl in GetVariablesAndParameters(iaie.Target))
                    yield return decl;
            }
            else if (expr is IMethodInvokeExpression imie)
            {
                foreach (IExpression arg in imie.Arguments)
                {
                    foreach (var decl in GetVariablesAndParameters(arg))
                        yield return decl;
                }
            }
            else if (expr is IBinaryExpression ibe)
            {
                foreach (var decl in GetVariablesAndParameters(ibe.Left))
                    yield return decl;
                foreach (var decl in GetVariablesAndParameters(ibe.Right))
                    yield return decl;
            }
            else if (expr is IUnaryExpression iue)
            {
                foreach (var decl in GetVariablesAndParameters(iue.Expression))
                    yield return decl;
            }
            else if (expr is IPropertyIndexerExpression ipie)
            {
                foreach (IExpression index in ipie.Indices)
                    foreach (var decl in GetVariablesAndParameters(index))
                        yield return decl;
                foreach (var decl in GetVariablesAndParameters(ipie.Target))
                    yield return decl;
            }
            else if (expr is IObjectCreateExpression ioce)
            {
                foreach (IExpression arg in ioce.Arguments)
                    foreach (var decl in GetVariablesAndParameters(arg))
                        yield return decl;
            }
            else if (expr is IAssignExpression iae)
            {
                foreach (var decl in GetVariablesAndParameters(iae.Expression))
                    yield return decl;
                foreach (var decl in GetVariablesAndParameters(iae.Target))
                    yield return decl;
            }
            else
            {
                object decl = GetDeclaration(expr);
                if (decl != null)
                    yield return decl;
            }
        }

        public IEnumerable<IArgumentReferenceExpression> GetArgumentReferenceExpressions(IExpression expr)
        {
            if (expr is IArgumentReferenceExpression are)
                yield return are;
            else if (expr is IArrayIndexerExpression iaie)
            {
                foreach (IExpression index in iaie.Indices)
                    foreach (var iare in GetArgumentReferenceExpressions(index))
                        yield return iare;
                foreach (var iare in GetArgumentReferenceExpressions(iaie.Target))
                    yield return iare;
            }
            else if (expr is IUnaryExpression iue)
            {
                foreach (var iare in GetArgumentReferenceExpressions(iue.Expression))
                    yield return iare;
            }
            else if (expr is IBinaryExpression ibe)
            {
                foreach (var iare in GetArgumentReferenceExpressions(ibe.Left))
                    yield return iare;
                foreach (var iare in GetArgumentReferenceExpressions(ibe.Right))
                    yield return iare;
            }
        }

        public IEnumerable<IExpression> GetConditionAndTargetIndexExpressions(IStatement stmt)
        {
            if (stmt is IConditionStatement ics)
            {
                yield return ics.Condition;
                foreach (var expr in GetConditionAndTargetIndexExpressions(ics.Then))
                    yield return expr;
            }
            else if (stmt is IForStatement ifs)
            {
                foreach (var expr in GetConditionAndTargetIndexExpressions(ifs.Body))
                    yield return expr;
            }
            else if (stmt is IBlockStatement ibs)
            {
                foreach (IStatement ist in ibs.Statements)
                {
                    foreach (var expr in GetConditionAndTargetIndexExpressions(ist))
                        yield return expr;
                }
            }
            else if (stmt is IExpressionStatement ies)
            {
                if (ies.Expression is IAssignExpression iae)
                {
                    // target indices are considered "conditions" for this purpose
                    foreach (var index in GetFlattenedIndices(iae.Target))
                        yield return index;
                }
            }
        }

        /// <summary>
        /// Get every expression on the lhs of an assignment.
        /// </summary>
        /// <param name="ist"></param>
        public IEnumerable<IExpression> GetTargets(IStatement ist)
        {
            if (ist is IExpressionStatement ies)
            {
                IExpression expr = ies.Expression;
                if (expr is IAssignExpression)
                {
                    IAssignExpression iae = (IAssignExpression)ies.Expression;
                    yield return iae.Target;
                }
                else if (expr is IVariableDeclarationExpression)
                {
                    yield return expr;
                }
            }
            else if (ist is IConditionStatement ics)
            {
                foreach (IStatement st in ics.Then.Statements)
                {
                    foreach (var expr in GetTargets(st))
                        yield return expr;
                }
            }
            else if (ist is IForStatement ifs)
            {
                foreach (IStatement st in ifs.Body.Statements)
                {
                    foreach (var expr in GetTargets(st))
                        yield return expr;
                }
            }
            else if (ist is IWhileStatement iws)
            {
                foreach (IStatement st in iws.Body.Statements)
                {
                    foreach (var expr in GetTargets(st))
                        yield return expr;
                }
            }
        }

        /// <summary>
        /// Apply action to every variable on the lhs of an assignment.
        /// </summary>
        /// <param name="ist"></param>
        public IEnumerable<IVariableDeclaration> GetTargetVariables(IStatement ist)
        {
            foreach (IExpression target in GetTargets(ist))
            {
                IVariableDeclaration ivd = GetVariableDeclaration(target);
                if (ivd != null)
                    yield return ivd;
            }
        }

        /// <summary>
        /// Extracts the variable declaration from a declaration statement.
        /// Returns null if the statement is not a declaration statement.
        /// </summary>
        /// <param name="ist"></param>
        /// <returns></returns>
        public IVariableDeclaration GetVariableDeclaration(IStatement ist)
        {
            if (ist is IExpressionStatement ies)
            {
                if (ies.Expression is IVariableDeclarationExpression ivde)
                    return ivde.Variable;
                else if (ies.Expression is IAssignExpression iae)
                {
                    if (iae.Target is IVariableDeclarationExpression ivde2)
                        return ivde2.Variable;
                }
            }
            return null;
        }

        /// <summary>
        /// Returns the number of indexing brackets at the end of expr (zero if none)
        /// </summary>
        /// <param name="iexpr"></param>
        /// <returns></returns>
        public int GetIndexingDepth(IExpression iexpr)
        {
            if (iexpr is IArrayIndexerExpression iaie)
                return 1 + GetIndexingDepth(iaie.Target);
            else
                return 0;
        }

        /// <summary>
        /// Add the declarations of all loop variables in expr to indVars.
        /// </summary>
        /// <param name="context"></param>
        /// <param name="indVars"></param>
        /// <param name="expr"></param>
        public void AddIndexers(BasicTransformContext context, List<IVariableDeclaration[]> indVars, IExpression expr)
        {
            if (expr is IVariableReferenceExpression)
                return;
            if (expr is IVariableDeclarationExpression)
                return;
            if (expr is IArgumentReferenceExpression)
                return;
            if (expr is IArrayIndexerExpression iaie)
            {
                IVariableDeclaration[] vars = new IVariableDeclaration[iaie.Indices.Count];
                for (int i = 0; i < vars.Length; i++)
                {
                    IVariableDeclaration ivd = GetVariableDeclaration(iaie.Indices[i]);
                    if (ivd != null && GetLoopForVariable(context, ivd) != null)
                        vars[i] = ivd;
                }
                AddIndexers(context, indVars, iaie.Target);
                indVars.Add(vars);
                return;
            }
            throw new NotImplementedException("Unsupported expression type in AddIndexers(): " + expr.GetType());
        }

        private IEnumerable<IExpression> GetFlattenedIndices(IExpression expr)
        {
            if (expr is IArrayIndexerExpression iaie)
            {
                foreach (IExpression index in GetFlattenedIndices(iaie.Target))
                    yield return index;
                foreach (IExpression index in iaie.Indices)
                    yield return index;
            }
        }

        /// <summary>
        /// Get a list of all index expressions at the end of expr, innermost first.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns>A list of indexing brackets, where each bracket is a collection of index expressions</returns>
        public List<IList<IExpression>> GetIndices(IExpression expr)
        {
            IExpression target;
            return GetIndices(expr, out target);
        }

        /// <summary>
        /// Get a list of all index expressions at the end of expr, innermost first.
        /// </summary>
        /// <param name="expr"></param>
        /// <param name="target">On exit, the innermost expression being indexed, i.e. the array variable</param>
        /// <returns>A list of indexing brackets, where each bracket is a collection of index expressions</returns>
        public List<IList<IExpression>> GetIndices(IExpression expr, out IExpression target)
        {
            List<IList<IExpression>> indices = new List<IList<IExpression>>();
            target = ForEachIndexingBracket(expr, indices.Add);
            return indices;
        }

        /// <summary>
        /// Apply action to all indexing brackets at the end of expr, innermost first.
        /// </summary>
        /// <param name="expr">The expression</param>
        /// <param name="action"></param>
        /// <returns>The innermost expression</returns>
        private IExpression ForEachIndexingBracket(IExpression expr, Action<IList<IExpression>> action)
        {
            if (expr is IArrayIndexerExpression iaie)
            {
                var target = ForEachIndexingBracket(iaie.Target, action);
                action(iaie.Indices);
                return target;
            }
            else
                return expr;
        }

        /// <summary>
        /// Remove the last index from an expression
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        public IExpression RemoveLastIndex(IExpression expr)
        {
            if (expr is IArrayIndexerExpression iaie)
            {
                if (iaie.Target is IArrayIndexerExpression)
                {
                    IArrayIndexerExpression aie = Builder.ArrayIndxrExpr();
                    aie.Indices.AddRange(iaie.Indices);
                    aie.Target = RemoveLastIndex(iaie.Target);
                    return aie;
                }
                else return iaie.Target;
            }
            else
            {
                return expr;
            }
        }

        /// <summary>
        /// Returns the for statement associated with the specified loop counter or null if none
        /// </summary>
        /// <param name="context"></param>
        /// <param name="loopRef"></param>
        /// <returns></returns>
        internal IForStatement GetLoopForVariable(BasicTransformContext context, IVariableReferenceExpression loopRef)
        {
            return GetLoopForVariable(context, GetVariableDeclaration(loopRef));
        }

        internal IForStatement GetLoopForVariable(BasicTransformContext context, IVariableDeclaration ivd, object excludeAncestor = null)
        {
            foreach (TransformInfo ti in context.InputStack)
            {
                if (ti.inputElement == excludeAncestor)
                    break;
                if (ti.inputElement is IForStatement loop)
                {
                    IVariableDeclaration loopVd = LoopVariable(loop);
                    if (ivd.Name == loopVd.Name)
                        return loop;
                }
            }
            return null;
        }

        public bool IsForwardLoop(IForStatement ifs)
        {
            if (ifs.Increment is IExpressionStatement ies)
            {
                if (ies.Expression is IAssignExpression iae)
                {
                    if (ChannelTransform.RemoveCast(iae.Expression) is IBinaryExpression ibe)
                    {
                        if (ibe.Operator == BinaryOperator.Add)
                        {
                            if (ibe.Right is ILiteralExpression ile)
                            {
                                if (ile.Value is int i)
                                    return (i >= 0);
                            }
                        }
                        else if (ibe.Operator == BinaryOperator.Subtract)
                        {
                            if (ibe.Right is ILiteralExpression ile)
                            {
                                if (ile.Value is int i)
                                    return (i < 0);
                            }
                        }
                    }
                }
                else if (ies.Expression is IUnaryExpression iue)
                {
                    return (iue.Operator == UnaryOperator.PostIncrement) || (iue.Operator == UnaryOperator.PreIncrement);
                }
            }
            throw new Exception("Unexpected loop increment");
        }


        // Convert a for loop of the form:
        //    for (int i = 0; i < N; i++) { ... }
        // to
        //    for (int i = N - 1; i >= 0; i--) { ... }
        // and vice versa.
        public void ReverseLoopDirection(IForStatement loop)
        {
            // For now, make strong assumption that the for loop is of standard form:
            //   for (<var> = <literal>; <var> <comp> <literal>; <var> <increment/decrement>)
            if (!(loop.Condition is IBinaryExpression))
                throw new ArgumentException("Loop condition is not a binary expression: " + loop.Condition);
            IBinaryExpression condition = (IBinaryExpression)loop.Condition;
            IExpressionStatement increment = (IExpressionStatement)loop.Increment;
            if (!(increment.Expression is IUnaryExpression))
            {
                throw new ArgumentException("Cannot reverse a for loop with increment: " + increment);
            }
            IUnaryExpression incrementExpr = (IUnaryExpression)increment.Expression;
            UnaryOperator unaryOp;
            if (incrementExpr.Operator == UnaryOperator.PostIncrement || incrementExpr.Operator == UnaryOperator.PreIncrement)
                unaryOp = UnaryOperator.PostDecrement;
            else if (incrementExpr.Operator == UnaryOperator.PostDecrement || incrementExpr.Operator == UnaryOperator.PreDecrement)
                unaryOp = UnaryOperator.PostIncrement;
            else
            {
                throw new ArgumentException("Cannot reverse a for loop with increment operator " + incrementExpr.Operator.ToString());
            }
            loop.Increment = Builder.ExprStatement(Builder.UnaryExpr(unaryOp, incrementExpr.Expression));
            if (!condition.Left.Equals(incrementExpr.Expression))
                throw new ArgumentException("Loop condition does not have loop variable on the left");
            IExpressionStatement initializer = (IExpressionStatement)loop.Initializer;
            IAssignExpression initAssignExpr = (IAssignExpression)initializer.Expression;
            IExpression initializationExpression = initAssignExpr.Expression;

            if (condition.Operator == BinaryOperator.LessThan)
            {
                // Construct a new binary expression that takes the old condition RHS and subtracts one.
                // This becomes the new initializer.
                IBinaryExpression newInitializationExpression = Builder.BinaryExpr(condition.Right, BinaryOperator.Subtract, Builder.LiteralExpr(1));
                loop.Initializer = Builder.AssignStmt(initAssignExpr.Target, newInitializationExpression);
            }
            else if (condition.Operator == BinaryOperator.LessThanOrEqual ||
                condition.Operator == BinaryOperator.GreaterThanOrEqual)
            {
                loop.Initializer = Builder.AssignStmt(initAssignExpr.Target, condition.Right);
            }
            else if (condition.Operator == BinaryOperator.GreaterThan)
            {
                IBinaryExpression newInitializationExpression = Builder.BinaryExpr(condition.Right, BinaryOperator.Add, Builder.LiteralExpr(1));
                loop.Initializer = Builder.AssignStmt(initAssignExpr.Target, newInitializationExpression);
            }
            else
            {
                throw new ArgumentException("Loop condition is not reversible: " + condition);
            }
            if (condition.Operator == BinaryOperator.LessThan ||
                condition.Operator == BinaryOperator.LessThanOrEqual)
            {
                loop.Condition = Builder.BinaryExpr(condition.Left, BinaryOperator.GreaterThanOrEqual, initializationExpression);
            }
            else
            {
                IBinaryExpression ibe = (IBinaryExpression)initializationExpression;
                if (ibe.Operator != BinaryOperator.Subtract ||
                   !ibe.Right.Equals(Builder.LiteralExpr(1)))
                    throw new ArgumentException("Initializer expression is not a subtraction: " + initializationExpression);
                loop.Condition = Builder.BinaryExpr(condition.Left, BinaryOperator.LessThan, ibe.Left);
            }
        }

        /// <summary>
        /// Returns the number of nested for loops in the deepest part of the statement.
        /// </summary>
        /// <param name="ist"></param>
        /// <returns></returns>
        public int ForLoopDepth(IStatement ist)
        {
            if (ist is IForStatement ifs)
            {
                return 1 + ForLoopDepth(ifs.Body);
            }
            else if (ist is IBlockStatement ibs)
            {
                int maxLoopDepth = 0;
                foreach (var st in ibs.Statements)
                {
                    int stDepth = ForLoopDepth(st);
                    if (stDepth > maxLoopDepth)
                    {
                        maxLoopDepth = stDepth;
                    }
                }
                return maxLoopDepth;
            }
            else if (ist is IConditionStatement ics)
            {
                return ForLoopDepth(ics.Then);
            }
            else
            {
                return 0;
            }
        }


        public IExpressionStatement FirstExpressionStatement(IStatement ist)
        {
            if (ist is IForStatement ifs)
            {
                foreach (var st in ifs.Body.Statements)
                {
                    if (!(st is ICommentStatement))
                    {
                        return FirstExpressionStatement(st);
                    }
                }
            }
            else if (ist is IConditionStatement ics)
            {
                foreach (var st in ics.Then.Statements)
                {
                    if (!(st is ICommentStatement))
                    {
                        return FirstExpressionStatement(st);
                    }
                }
            }
            else if (ist is IExpressionStatement ies)
            {
                return ies;
            }
            return null;
            //throw new Exception("Didn't find expression statement");
        }

        // Convert < to >=, <= to >, == to !=, etc.
        public bool TryNegateOperator(BinaryOperator op, out BinaryOperator negatedOp)
        {
            BinaryOperator[,] pairs = new BinaryOperator[,] {
            { BinaryOperator.LessThanOrEqual, BinaryOperator.GreaterThan },
            { BinaryOperator.GreaterThanOrEqual, BinaryOperator.LessThan },
            { BinaryOperator.ValueEquality, BinaryOperator.ValueInequality },
            { BinaryOperator.IdentityEquality, BinaryOperator.IdentityInequality }
        };

            for (int i = 0; i < pairs.GetLength(0); i++)
            {
                if (op == pairs[i, 0])
                {
                    negatedOp = pairs[i, 1];
                    return true;
                }
                else if (op == pairs[i, 1])
                {
                    negatedOp = pairs[i, 0];
                    return true;
                }
            }
            negatedOp = default(BinaryOperator);
            return false;
        }

        /// <summary>
        /// Returns true if any of the expression are stochastic.
        /// </summary>
        /// <param name="context"></param>
        /// <param name="iec"></param>
        /// <returns></returns>
        internal static bool IsAnyStochastic(BasicTransformContext context, IList<IExpression> iec)
        {
            foreach (IExpression expr in iec) if (IsStochastic(context, expr)) return true;
            return false;
        }

        /// <summary>
        /// Returns true if the expression is stochastic.
        /// </summary>
        /// <param name="context"></param>
        /// <param name="expr"></param>
        /// <returns></returns>
        internal static bool IsStochastic(BasicTransformContext context, IExpression expr)
        {
            if (expr is ILiteralExpression) return false;
            if (expr is IDefaultExpression) return false;
            if (expr is IMethodInvokeExpression imie)
            {
                bool stochArgs = IsAnyStochastic(context, imie.Arguments);
                bool stochFactor = false;
                FactorManager.FactorInfo info = GetFactorInfo(context, imie);
                if (info != null) stochFactor = !info.IsDeterministicFactor;
                bool st = stochArgs || stochFactor;
                //if (st) context.OutputAttributes.Set(imie, new Stochastic()); // for IfCuttingTransform
                return st;
            }
            if (expr is IArgumentReferenceExpression) return false;
            if (expr is IPropertyReferenceExpression ipre) return IsStochastic(context, ipre.Target);
            if (expr is IFieldReferenceExpression ifre) return IsStochastic(context, ifre.Target);
            if (expr is IArrayCreateExpression) return false;
            if (expr is IObjectCreateExpression ioce)
            {
                return IsAnyStochastic(context, ioce.Arguments);
            }
            if (expr is IUnaryExpression iue)
            {
                return IsStochastic(context, iue.Expression);
            }
            if (expr is IBinaryExpression ibe)
            {
                return IsStochastic(context, ibe.Left) || IsStochastic(context, ibe.Right);
            }
            if (expr is IArrayIndexerExpression iaie)
            {
                return IsStochastic(context, iaie.Target) || IsAnyStochastic(context, iaie.Indices);
            }
            if (expr is ICastExpression ice) return IsStochastic(context, ice.Expression);
            if (expr is ICheckedExpression ichecked) return IsStochastic(context, ichecked.Expression);
            if (expr is IPropertyIndexerExpression ipie)
            {
                return IsStochastic(context, ipie.Target) || IsAnyStochastic(context, ipie.Indices);
            }
            if (expr is IAddressDereferenceExpression) return false;
            if (expr is IAddressOutExpression) return false;
            if (expr is ILambdaExpression) return false; // todo: stochastic case?
            if (expr is IAnonymousMethodExpression) return false;
            if (expr is ITypeOfExpression) return false;
            if (expr is IMethodReferenceExpression) return false;

            IVariableDeclaration ivd = Instance.GetVariableDeclaration(expr);
            if (ivd == null)
            {
                context.Error("Could not find stochasticity of expression of type " + expr.GetType().Name + ": " + expr);
                return false;
            }
            return IsStochastic(context, ivd);
        }

        internal static bool IsStochastic(BasicTransformContext context, IVariableDeclaration ivd)
        {
            VariableInformation vi = VariableInformation.GetVariableInformation(context, ivd);
            return vi.IsStochastic;
        }

        /// <summary>
        /// Returns true if any of the expression are stochastic.
        /// </summary>
        /// <param name="context"></param>
        /// <param name="iec"></param>
        /// <returns></returns>
        internal static bool AnyNeedsMarginalDividedByPrior(BasicTransformContext context, IList<IExpression> iec)
        {
            foreach (IExpression expr in iec) if (NeedsMarginalDividedByPrior(context, expr)) return true;
            return false;
        }

        /// <summary>
        /// Returns true if the expression is stochastic.
        /// </summary>
        /// <param name="context"></param>
        /// <param name="expr"></param>
        /// <returns></returns>
        internal static bool NeedsMarginalDividedByPrior(BasicTransformContext context, IExpression expr)
        {
            if (expr is ILiteralExpression) return false;
            if (expr is IDefaultExpression) return false;
            if (expr is IMethodInvokeExpression imie)
            {
                bool stochArgs = AnyNeedsMarginalDividedByPrior(context, imie.Arguments);
                bool stochFactor = false;
                FactorManager.FactorInfo info = GetFactorInfo(context, imie);
                if (info != null) stochFactor = !info.IsDeterministicFactor;
                bool st = stochArgs || stochFactor;
                //if (st) context.OutputAttributes.Set(imie, new Stochastic()); // for IfCuttingTransform
                return st;
            }
            if (expr is IArgumentReferenceExpression) return false;
            if (expr is IPropertyReferenceExpression ipre) return NeedsMarginalDividedByPrior(context, ipre.Target);
            if (expr is IFieldReferenceExpression ifre) return NeedsMarginalDividedByPrior(context, ifre.Target);
            if (expr is IArrayCreateExpression) return false;
            if (expr is IObjectCreateExpression ioce)
            {
                return AnyNeedsMarginalDividedByPrior(context, ioce.Arguments);
            }
            if (expr is IUnaryExpression iue)
            {
                return NeedsMarginalDividedByPrior(context, iue.Expression);
            }
            if (expr is IBinaryExpression ibe)
            {
                return NeedsMarginalDividedByPrior(context, ibe.Left) || NeedsMarginalDividedByPrior(context, ibe.Right);
            }
            if (expr is IArrayIndexerExpression iaie)
            {
                return NeedsMarginalDividedByPrior(context, iaie.Target) || AnyNeedsMarginalDividedByPrior(context, iaie.Indices);
            }
            if (expr is ICastExpression ice) return NeedsMarginalDividedByPrior(context, ice.Expression);
            if (expr is ICheckedExpression ichecked) return NeedsMarginalDividedByPrior(context, ichecked.Expression);
            if (expr is IPropertyIndexerExpression ipie)
            {
                return NeedsMarginalDividedByPrior(context, ipie.Target) || AnyNeedsMarginalDividedByPrior(context, ipie.Indices);
            }
            if (expr is IAddressDereferenceExpression) return false;
            if (expr is IAddressOutExpression) return false;
            if (expr is ILambdaExpression) return false; // todo: stochastic case?
            if (expr is IAnonymousMethodExpression) return false;
            if (expr is ITypeOfExpression) return false;
            if (expr is IMethodReferenceExpression) return false;

            IVariableDeclaration ivd = Instance.GetVariableDeclaration(expr);
            if (ivd == null)
            {
                context.Error("Could not find stochasticity of expression of type " + expr.GetType().Name + ": " + expr);
                return false;
            }
            return NeedsMarginalDividedByPrior(context, ivd);
        }

        internal static bool NeedsMarginalDividedByPrior(BasicTransformContext context, IVariableDeclaration ivd)
        {
            VariableInformation vi = VariableInformation.GetVariableInformation(context, ivd);
            return vi.NeedsMarginalDividedByPrior;
        }

        internal static bool IsInfer(IExpression expr)
        {
            return Instance.IsStaticGenericMethod(expr, new Action<object>(InferNet.Infer)) ||
                   Instance.IsStaticGenericMethod(expr, new Action<object, string>(InferNet.Infer)) ||
                   Instance.IsStaticGenericMethod(expr, new Action<object, string, QueryType>(InferNet.Infer));
        }

        internal static bool IsIsIncreasing(IExpression expr)
        {
            return Instance.IsStaticMethod(expr, new Func<int,bool>(InferNet.IsIncreasing));
        }

        internal static FactorManager.FactorInfo GetFactorInfo(BasicTransformContext context, IMethodInvokeExpression imie)
        {
            if (!context.InputAttributes.Has<FactorManager.FactorInfo>(imie))
            {
                if (Instance.IsStaticMethod(imie, typeof(InferNet))) return null;
                MethodInfo methodInfo = (MethodInfo)Builder.ToMethodThrows(imie.Method.Method);
                if (methodInfo == null) return null;
                FactorManager.FactorInfo info = FactorManager.GetFactorInfo(methodInfo);
                context.InputAttributes.Set(imie, info);
            }
            return context.InputAttributes.Get<FactorManager.FactorInfo>(imie);
        }

        /// <summary>
        /// Removes a cast when it is safe to do so.
        /// </summary>
        /// <param name="expr"></param>
        /// <returns></returns>
        internal static IExpression RemoveCast(IExpression expr)
        {
            // used to remove spurious casts
            if (expr is ICastExpression ice)
            {
                if (expr.GetExpressionType().IsAssignableFrom(ice.Expression.GetExpressionType()))
                {
                    return ice.Expression;
                }
            }
            return expr;
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}