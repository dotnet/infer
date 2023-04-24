// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using System.IO;
using System.Reflection;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Inserts statements that write all messages to csv files (at the end of each iteration).
    /// </summary>
    internal class TracingTransform : ShallowCopyTransform
    {
        IVariableDeclaration iterationVar;
        readonly Dictionary<Set<IVariableDeclaration>, TableInfo> tableOfIndexVars = new Dictionary<Set<IVariableDeclaration>, TableInfo>();
        MethodInfo writeMethod, writeBytesMethod, writeLineMethod, flushMethod, disposeMethodInfo;
        IMethodDeclaration traceWriterMethod, disposeMethod;
        public static bool UseToString = true;

        public override string Name
        {
            get
            {
                return "TracingTransform";
            }
        }

        protected override IStatement ConvertFor(IForStatement ifs)
        {
            bool isIterationLoop = false;
            var loopVar = Recognizer.LoopVariable(ifs);
            if (loopVar.Name == "iteration")
            {
                iterationVar = loopVar;
                tableOfIndexVars.Clear();
                isIterationLoop = true;
            }
            IStatement ist = base.ConvertFor(ifs);
            if (isIterationLoop)
            {
                IForStatement fs = (IForStatement)ist;
                foreach (var table in tableOfIndexVars.Values) WriteTable(table, fs.Body.Statements);
                iterationVar = null;
            }
            return ist;
        }

        void MakeTraceWriterMethod()
        {
            ITypeDeclaration td = context.FindOutputForAncestor<ITypeDeclaration, ITypeDeclaration>();
            // StreamWriter MakeWriter(string name, string header)
            IParameterDeclaration nameDecl = Builder.Param("name", typeof(string));
            IParameterDeclaration headerDecl = Builder.Param("header", typeof(string));
            traceWriterMethod = Builder.MethodDecl(MethodVisibility.Public, "TraceWriter", typeof(StreamWriter), td, nameDecl, headerDecl);
            traceWriterMethod.Static = true;
            var name = Builder.ParamRef(nameDecl);
            var header = Builder.ParamRef(headerDecl);
            var stmts = traceWriterMethod.Body.Statements;
            string folder = td.Name;
            stmts.Add(Builder.ExprStatement(Builder.StaticMethod(new Func<string, DirectoryInfo>(Directory.CreateDirectory), Builder.LiteralExpr(folder))));
            IExpression pathExpr = Builder.BinaryExpr(BinaryOperator.Add, name, Builder.LiteralExpr(UseToString ? ".tsv" : ".csv"));
            pathExpr = Builder.BinaryExpr(BinaryOperator.Add, Builder.LiteralExpr(folder + "/"), pathExpr);
            var writerDecl = Builder.VarDecl("writer", typeof(StreamWriter));
            var ctorExpr = Builder.NewObject(typeof(StreamWriter), pathExpr);
            stmts.Add(Builder.AssignStmt(Builder.VarDeclExpr(writerDecl), ctorExpr));
            var writer = Builder.VarRefExpr(writerDecl);
            // writer.BaseStream.Write(Encoding.UTF8.GetPreamble(), 0, Encoding.UTF8.GetPreamble().Length);
            Type encodingType = typeof(System.Text.Encoding);
            IExpression preambleExpr = Builder.Method(Builder.StaticPropRefExpr(encodingType, "UTF8"), 
                new Func<byte[]>(System.Text.Encoding.UTF8.GetPreamble));
            IExpression baseStream = Builder.PropRefExpr(writer, typeof(StreamWriter), "BaseStream");
            stmts.Add(GetWriteBytesStatement(baseStream, preambleExpr));
            stmts.Add(GetWriteStatement(writer, header));
            stmts.Add(GetWriteLineStatement(writer));
            stmts.Add(Builder.Return(writer));
            td.Methods.Add(traceWriterMethod);
        }

        void MakeDisposeMethod()
        {
            ITypeDeclaration td = context.FindOutputForAncestor<ITypeDeclaration, ITypeDeclaration>();
            td.Interfaces.Add((ITypeReference)Builder.TypeRef(typeof(IDisposable)));
            disposeMethod = Builder.MethodDecl(MethodVisibility.Public, "Dispose", typeof(void), td);
            td.Methods.Add(disposeMethod);
        }

        void WriteTable(TableInfo table, ICollection<IStatement> output)
        {
            string writerName = VariableInformation.GenerateName(context, "_writer");
            ITypeDeclaration td = context.FindOutputForAncestor<ITypeDeclaration, ITypeDeclaration>();
            IFieldDeclaration fd = Builder.FieldDecl(writerName, typeof(StreamWriter), td);
            fd.Documentation = "Writes "+table.Name;
            context.AddMember(fd);
            td.Fields.Add(fd);
            var writer = Builder.FieldRefExpr(fd);
            IForStatement innerForStatement;
            IForStatement fs = Builder.NestedForStmt(table.indexVars, table.sizes, out innerForStatement);
            ICollection<IStatement> output2;
            if (fs != null)
            {
                output.Add(fs);
                output2 = innerForStatement.Body.Statements;
            }
            else
            {
                output2 = output;
            }
            StringBuilder header = new StringBuilder();
            header.Append("iteration");
            output2.Add(GetWriteStatement(writer, Builder.VarRefExpr(iterationVar)));
            string delimiter = UseToString ? "\t" : ",";
            var writeDelimiter = GetWriteStatement(writer, Builder.LiteralExpr(delimiter));
            foreach (var indexVar in table.indexVars)
            {
                header.Append(delimiter);
                header.Append(indexVar.Name);
                output2.Add(writeDelimiter);
                output2.Add(GetWriteStatement(writer, Builder.VarRefExpr(indexVar)));
            }
            foreach (var messageBaseExpr in table.messageExprs)
            {
                Stack<IExpression> conditions = new Stack<IExpression>();
                object messageVar = Recognizer.GetDeclaration(messageBaseExpr);
                VariableInformation varInfo = VariableInformation.GetVariableInformation(context, messageVar);
                IExpression messageExpr = messageBaseExpr;
                foreach (var bracket in varInfo.indexVars)
                {
                    if (!messageExpr.GetExpressionType().IsValueType)
                        conditions.Push(Builder.BinaryExpr(BinaryOperator.IdentityEquality, messageExpr, Builder.LiteralExpr(null)));
                    var indices = Util.ArrayInit(bracket.Length, i => Builder.VarRefExpr(bracket[i]));
                    messageExpr = Builder.ArrayIndex(messageExpr, indices);
                }
                if (!messageExpr.GetExpressionType().IsValueType)
                    conditions.Push(Builder.BinaryExpr(BinaryOperator.IdentityEquality, messageExpr, Builder.LiteralExpr(null)));
                if (messageExpr.GetExpressionType().IsPrimitive)
                {
                    header.Append(delimiter);
                    header.Append(varInfo.Name);
                    output2.Add(writeDelimiter);
                    output2.Add(GetWriteStatement(writer, AddConditions(messageExpr, conditions)));
                }
                else
                {
                    Dictionary<string, IExpression> dict = GetProperties(messageExpr);
                    foreach (var entry in dict)
                    {
                        header.Append(delimiter);
                        header.Append(varInfo.Name + entry.Key);
                        output2.Add(writeDelimiter);
                        output2.Add(GetWriteStatement(writer, AddConditions(entry.Value, conditions)));
                    }
                }
            }
            output2.Add(GetWriteLineStatement(writer));
            output.Add(GetFlushStatement(writer));
            if (traceWriterMethod == null) MakeTraceWriterMethod();
            fd.Initializer = Builder.StaticMethod(traceWriterMethod, Builder.LiteralExpr(table.Name), Builder.LiteralExpr(header.ToString()));
            if (disposeMethod == null) MakeDisposeMethod();
            disposeMethod.Body.Statements.Add(GetDisposeStatement(writer));
        }

        Dictionary<string, IExpression> GetProperties(IExpression expr)
        {
            Dictionary<string, IExpression> dict = new Dictionary<string, IExpression>();
            Type type = expr.GetExpressionType();
            if (UseToString)
            {
                var toStringMethod = type.GetMethod("ToString", new Type[0]);
                dict["ToString"] = Builder.Method(expr, toStringMethod);
            }
            else if (typeof(Distributions.Gaussian).IsAssignableFrom(type))
            {
                foreach (var field in new[] { "MeanTimesPrecision", "Precision" })
                {
                    dict[field] = Builder.FieldRefExpr(expr, type, field);
                }
            }
            else
            {
                Type[] faces = type.GetInterfaces();
                bool hasGetMean = false;
                bool hasGetVariance = false;
                foreach (Type face in faces)
                {
                    if (face.Name == "CanGetMean`1")
                        hasGetMean = true;
                    else if (face.Name == "CanGetVariance`1")
                        hasGetVariance = true;
                }
                if (hasGetMean)
                {
                    var meanMethod = type.GetMethod("GetMean", new Type[0]);
                    dict["Mean"] = Builder.Method(expr, meanMethod);
                }
                if (hasGetVariance)
                {
                    var varianceMethod = type.GetMethod("GetVariance", new Type[0]);
                    dict["Variance"] = Builder.Method(expr, varianceMethod);
                }
            }
            return dict;
        }

        IExpression AddConditions(IExpression expr, Stack<IExpression> conditions)
        {
            if (conditions.Count > 0)
            {
                var toStringMethod = expr.GetExpressionType().GetMethod("ToString", new Type[0]);
                expr = Builder.Method(expr, Builder.MethodRef(toStringMethod));
            }
            foreach (IExpression condition in conditions)
            {
                var condExpr = Builder.CondExpr();
                condExpr.Condition = condition;
                condExpr.Then = Builder.LiteralExpr("");
                condExpr.Else = expr;
                expr = condExpr;
            }
            return expr;
        }

        IExpressionStatement GetWriteBytesStatement(IExpression writer, IExpression bufferExpr)
        {
            if (writeBytesMethod == null)
            {
                writeBytesMethod = typeof(Stream).GetMethod("Write", new Type[] { typeof(byte[]), typeof(int), typeof(int) });
            }
            var lengthExpr = Builder.PropRefExpr(bufferExpr, typeof(byte[]), "Length");
            var invokeExpr = Builder.Method(writer, writeBytesMethod, bufferExpr, Builder.LiteralExpr(0), lengthExpr);
            return Builder.ExprStatement(invokeExpr);
        }

        IExpressionStatement GetWriteStatement(IExpression writer, IExpression expr)
        {
            if (writeMethod == null)
            {
                writeMethod = typeof(StreamWriter).GetMethod("Write", new Type[] { typeof(object) });
            }
            var invokeExpr = Builder.Method(writer, writeMethod, expr);
            return Builder.ExprStatement(invokeExpr);
        }

        IExpressionStatement GetWriteLineStatement(IExpression writer)
        {
            if (writeLineMethod == null)
            {
                writeLineMethod = typeof(StreamWriter).GetMethod("WriteLine", new Type[0]);
            }
            var invokeExpr = Builder.Method(writer, writeLineMethod);
            return Builder.ExprStatement(invokeExpr);
        }

        IExpressionStatement GetFlushStatement(IExpression writer)
        {
            if (flushMethod == null)
            {
                flushMethod = typeof(StreamWriter).GetMethod("Flush", new Type[0]);
            }
            var invokeExpr = Builder.Method(writer, flushMethod);
            return Builder.ExprStatement(invokeExpr);
        }

        IExpressionStatement GetDisposeStatement(IExpression writer)
        {
            if (disposeMethodInfo == null)
            {
                disposeMethodInfo = typeof(StreamWriter).GetMethod("Dispose", new Type[0]);
            }
            var invokeExpr = Builder.Method(writer, disposeMethodInfo);
            return Builder.ExprStatement(invokeExpr);
        }

        void RegisterVariable(IExpression messageExpr)
        {
            // get the set of indexVars
            Set<IVariableDeclaration> indexVars = new Set<IVariableDeclaration>();
            object messageDecl = Recognizer.GetDeclaration(messageExpr);
            var varInfo = VariableInformation.GetVariableInformation(context, messageDecl);
            foreach (var bracket in varInfo.indexVars)
            {
                indexVars.AddRange(bracket);
            }
            TableInfo table;
            if (!tableOfIndexVars.TryGetValue(indexVars, out table))
            {
                table = new TableInfo();
                StringBuilder sb = new StringBuilder();
                foreach (IVariableDeclaration indexVar in indexVars)
                {
                    if (sb.Length > 0) sb.Append("_");
                    sb.Append(indexVar.Name);
                }
                if (sb.Length == 0) sb.Append("scalar");
                table.Name = VariableInformation.GenerateName(context, sb.ToString());
                for (int bracket = 0; bracket < varInfo.indexVars.Count; bracket++)
                {
                    for (int i = 0; i < varInfo.indexVars[bracket].Length; i++)
                    {
                        table.indexVars.Add(varInfo.indexVars[bracket][i]);
                        table.sizes.Add(varInfo.sizes[bracket][i]);
                    }
                }
                tableOfIndexVars.Add(indexVars, table);
            }
            table.messageExprs.Add(messageExpr);
        }

        protected override IExpression ConvertAssign(IAssignExpression iae)
        {
            if (iterationVar != null)
            {
                bool isOperatorStmt = false;
                foreach (var ist in context.FindAncestors<IStatement>())
                {
                    if (context.InputAttributes.Has<OperatorStatement>(ist))
                    {
                        isOperatorStmt = true;
                        break;
                    }
                }
                if (isOperatorStmt)
                {
                    // do not trace assignments to loop variables
                    var target = Recognizer.GetTarget(iae.Target);
                    var ivd = Recognizer.GetVariableDeclaration(target);
                    bool isLoopVar = (ivd != null && Recognizer.GetLoopForVariable(context, ivd) != null);
                    if(!isLoopVar)
                    {
                        RegisterVariable(target);
                    }
                }
            }
            return base.ConvertAssign(iae);
        }

        /// <summary>
        /// Describes the contents of a single csv file.
        /// </summary>
        class TableInfo
        {
            public string Name;
            public List<IVariableDeclaration> indexVars = new List<IVariableDeclaration>();
            public List<IExpression> sizes = new List<IExpression>();
            public Set<IExpression> messageExprs = new Set<IExpression>();
        }
    }
}
