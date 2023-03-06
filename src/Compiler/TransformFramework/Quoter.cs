// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Serialization;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;

namespace Microsoft.ML.Probabilistic.Compiler
{
    /// <summary>
    /// Provides static methods for quoting objects into instances of the code model.
    /// </summary>
    public static class Quoter
    {
        /// <summary>
        /// Helps build expressions
        /// </summary>
        private static CodeBuilder Builder = CodeBuilder.Instance;

        /// <summary>
        /// Quotes the object and returns the quoted expression.
        /// </summary>
        /// <param name="value">The object to quote</param>
        /// <returns>An expression which would evaluate to the supplied value.</returns>
        public static IExpression Quote(object value)
        {
            if (value == null) return Builder.LiteralExpr(value);
            IExpression expr;

            if (TryQuoteConstructable(value, out expr)) return expr;

            if (value is Distributions.Kernels.NNKernel g)
            {
                // todo: change this to use ConstructionAttribute
                expr = Builder.NewObject(value.GetType(), Quote(g.GetLogWeightVariances()), Quote(g.GetLogBiasWeightVariance()));
            }
            else if (value is Array array)
            {
                expr = QuoteArray(array);
            }
            else if (value is ConvertibleToArray convertibleToArray)
            {
                expr = Builder.NewObject(value.GetType(), QuoteArray(convertibleToArray.ToArray()));
            }
            else if (value is IList ilist)
            {
                expr = QuoteList(ilist);
            }
            else if (value is IDictionary idictionary)
            {
                expr = QuoteDictionary(idictionary);
            }
            else if (value is Enum)
            {
                expr = Builder.LiteralExpr(value);
            }
            else if (value is PropertyInfo propertyInfo)
            {
                expr = QuotePropertyInfo(propertyInfo);
            }
            else if (value is System.Linq.Expressions.Expression linqExpression)
            {
                var expconv = new Transforms.LinqExpressionTransform();
                expr = expconv.Convert(linqExpression);
            }
            else if (value is DateTime dt)
            {
                var args = new List<int> {dt.Year, dt.Month, dt.Day};
                if ((dt.Hour != 0) || (dt.Minute != 0) || (dt.Second != 0) || (dt.Millisecond != 0))
                {
                    args.Add(dt.Hour);
                    args.Add(dt.Minute);
                    args.Add(dt.Second);
                    if (dt.Millisecond != 0) args.Add(dt.Millisecond);
                }
                var argsArray = new IExpression[args.Count];
                for (int i = 0; i < args.Count; i++) argsArray[i] = Builder.LiteralExpr(args[i]);
                expr = Builder.NewObject(typeof (DateTime), argsArray);
            }
            else if (value is Delegate d)
            {
                expr = Builder.NewObject(d.GetType(), Builder.MethodRefExpr(Builder.MethodRef(d.Method), d.Target is null ? Builder.TypeRefExpr(d.Method.DeclaringType) : Quote(d.Target)));
            }
            else
            {
                Type t = value.GetType();
                if (t.IsGenericType)
                {
                    Type gt = t.GetGenericTypeDefinition();
                    if (gt.Equals(typeof (List<>)))
                    {
                        expr = (IExpression) Reflection.Invoker.InvokeStatic(typeof (Quoter), "QuoteList", value);
                    }
                    else if (gt == typeof (GibbsMarginal<,>))
                    {
                        expr = (IExpression) Reflection.Invoker.InvokeStatic(typeof (Quoter), "QuoteGibbsEstimator", value);
                    }
                }
                else
                    // See if this is an estimator
                    expr = QuoteEstimator(value);
                if (expr == null) expr = ExpressionEvaluator.Quote(value);
            }

            if (expr == null)
                throw new NotImplementedException("Cannot quote '" + value + "' - consider adding a ConstructionAttribute to the appropriate constructor of type " +
                                                  value.GetType() + ".");
            return expr;
        }

        /// <summary>
        /// Trys to quote an object using any Construction attributes that it is
        /// annotated with.
        /// </summary>
        /// <remarks>
        /// This method can be used to quote a different object that the one passed in, by using
        /// 'useWhenOverride'.  For example, setting it to 'IsUniform' will mean that the value will
        /// be quoted as if 'IsUniform' were true.  In general, this will result in quoting the closest
        /// value that meets the specified condition.
        /// </remarks>
        /// <param name="value">The value to quote</param>
        /// <param name="expr">The quoted expression or null if quoting failed</param>
        /// <param name="useWhenOverride">Overrides which construction attribute to use, to quote
        /// <param name="valueType">The type of the value to quote or null to use value.GetType()</param>
        /// the object passed in modified to have the condition be true</param>
        /// <returns>True if the quoting succeeded</returns>
        internal static bool TryQuoteConstructable(object value, out IExpression expr, string useWhenOverride = null, Type valueType = null)
        {
            if (valueType == null) valueType = value.GetType();
            // Automated quoting using the 'Construction' attribute
            var cas = ConstructionAttribute.GetConstructionAttribute(valueType);
            expr = null;
            foreach (var ca in cas)
            {
                if (useWhenOverride == null)
                {
                    if (!ca.IsApplicable(value)) continue;
                }
                else
                {
                    if (!Equals(ca.UseWhen, useWhenOverride)) continue;
                }
                IExpression[] pars = new IExpression[ca.Params == null ? 0 : ca.Params.Length];

                // If we don't have an object instance, we cannot quote
                // construction methods that take parameters.
                if ((pars.Length > 0) && (value == null)) return false;

                for (int i = 0; i < pars.Length; i++)
                {
                    Type type;
                    object paramValue = ca.GetParamValue(i, value, out type);
                    // Use default(T) instead of null, so that type information is preserved.
                    pars[i] = (paramValue == null) ? Builder.DefaultExpr(type) : Quote(paramValue);
                }
                if (ca.TargetMember is ConstructorInfo)
                {
                    // We need to use a constructor
                    expr = Builder.NewObject(value.GetType(), pars);
                }
                else
                {
                    // We need to use a factory method
                    expr = Builder.StaticMethod((MethodInfo) ca.TargetMember, pars);
                }
                return true;
            }
            return false;
        }

        /// <summary>
        /// When quoting objects whose types are not public, the quoted expression
        /// must have a different type.  If the passed in type is not public, this method 
        /// returns the public type that it will be quoted as.  If it is public, the type
        /// argument will be returned as is.
        /// </summary>
        /// <param name="tp"></param>
        /// <returns></returns>
        internal static Type GetPublicType(Type tp)
        {
            if (typeof (PropertyInfo).IsAssignableFrom(tp)) return typeof (PropertyInfo);
            return tp;
        }

        /// <summary>
        /// Whether constants of this type should be inlined as literals, rather that
        /// put in a single place.
        /// </summary>
        /// <param name="tp"></param>
        /// <returns></returns>
        internal static bool ShouldInlineType(Type tp)
        {
            return (tp.IsPrimitive || tp.IsEnum || tp == typeof (string));
        }

        private static IExpression QuotePropertyInfo(PropertyInfo propertyInfo)
        {
            // Make a typeof() expression
            var typeOfExpr = Builder.TypeOf(Builder.TypeRef(propertyInfo.DeclaringType));
            // Call GetProperty() on it, passing in the name
            var getProp = Builder.Method(typeOfExpr, typeof (Type).GetMethod("GetProperty", new Type[] {typeof (string)}), Builder.LiteralExpr(propertyInfo.Name));
            return getProp;
        }

        public static IExpression QuoteDictionary(IDictionary dictionary)
        {
            var newDict = Builder.NewObject(dictionary.GetType());
            var init = Builder.BlockExpr();
            foreach (DictionaryEntry entry in dictionary)
            {
                var entryBlock = Builder.BlockExpr();
                entryBlock.Expressions.Add(Quote(entry.Key));
                entryBlock.Expressions.Add(Quote(entry.Value));
                init.Expressions.Add(entryBlock);
            }
            newDict.Initializer = init;
            return newDict;
        }

        public static IExpression QuoteList(IList list)
        {
            var newList = Builder.NewObject(list.GetType());
            var be = Builder.BlockExpr();
            foreach (var obj in list)
            {
                be.Expressions.Add(Quote(obj));
            }
            if (be.Expressions.Count > 0)
            {
                newList.Initializer = be;
            }
            return newList;
        }

        public static IExpression[] QuoteItems<T>(ICollection<T> list)
        {
            IExpression[] result = new IExpression[list.Count];
            int i = 0;
            foreach (T item in list)
            {
                result[i++] = Quote(item);
            }
            return result;
        }

        // copied from ExpressionEvaluator.cs
        public static IExpression QuoteArray(Array array)
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

        // Quote an estimator expression
        public static IExpression QuoteEstimator(object value)
        {
            // todo: remove and use Construction attributes
            IExpression expr = null;
            if (value is BernoulliEstimator)
            {
                BernoulliEstimator g = (BernoulliEstimator) value;
                expr = Builder.NewObject(value.GetType());
            }
            else if (value is DirichletEstimator)
            {
                DirichletEstimator g = (DirichletEstimator) value;
                expr = Builder.NewObject(value.GetType(), Quote((g.Dimension)));
            }
            else if (value is DiscreteEstimator)
            {
                DiscreteEstimator g = (DiscreteEstimator) value;
                expr = Builder.NewObject(value.GetType(), Quote((g.Dimension)));
            }
            else if (value is GammaEstimator)
            {
                GammaEstimator g = (GammaEstimator) value;
                expr = Builder.NewObject(value.GetType());
            }
            else if (value is GaussianEstimator)
            {
                GaussianEstimator g = (GaussianEstimator) value;
                expr = Builder.NewObject(value.GetType());
            }
            else if (value is VectorGaussianEstimator)
            {
                VectorGaussianEstimator g = (VectorGaussianEstimator) value;
                expr = Builder.NewObject(value.GetType(), Quote(g.Dimension));
            }
            else if (value is WishartEstimator)
            {
                WishartEstimator g = (WishartEstimator) value;
                expr = Builder.NewObject(value.GetType(), Quote(g.Dimension));
            }

            return expr;
        }
    }
}