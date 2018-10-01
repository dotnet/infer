// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.Transforms;

namespace Microsoft.ML.Probabilistic
{
    public class Csoft
    {
        public static T Infer<T>(object variable)
        {
            throw new InvalidOperationException("This method must not be executed");
        }

        public static void Observe(object variable, object value)
        {
            throw new InvalidOperationException("This method must not be executed");
        }

        public static T GenerateSubclass<T>(InferenceEngine engine)
        {
            return (T)ProcessQueryClass(engine, typeof(T), null);
        }

        public static void ExecuteQuery(InferenceEngine engine, Action queryMethod)
        {
            var singleQuery = (ISingleQuery)ProcessQueryClass(engine, queryMethod.Method.DeclaringType, queryMethod.Method);
            singleQuery.ExecuteQuery();
        }

        public static object ProcessQueryClass(InferenceEngine engine, Type type, MethodInfo singleQuery)
        {
            var typeDecl = RoslynDeclarationProvider.GetTypeDeclaration(type, false);
            var queryTransform = new QueryTransform(engine, singleQuery);
            var compiledQueries = queryTransform.Transform(typeDecl);

            bool showCode = false;
            if (showCode)
            {
                var sw = new StringWriter();
                var lw = new CSharpWriter();
                var sn = lw.GenerateSource(compiledQueries);
                LanguageWriter.WriteSourceNode(sw, sn);
                var sourceCode = sw.ToString();
            }

            return engine.Compiler.CompileWithoutParams<object>(new[] { compiledQueries }.ToList());
        }

        public static IDeclarationProvider RoslynDeclarationProvider;
    }
}
