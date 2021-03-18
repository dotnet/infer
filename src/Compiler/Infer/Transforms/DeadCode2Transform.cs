// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Diagnostics;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler.Graphs;
using Microsoft.ML.Probabilistic.Utilities;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
    /// <summary>
    /// Removes assignments to variables whose value is never used before being overwritten.
    /// </summary>
    internal class DeadCode2Transform : ShallowCopyTransform
    {
        public override string Name
        {
            get
            {
                return "DeadCode2Transform";
            }
        }

        public static bool debug;
        readonly ModelCompiler compiler;
        LivenessAnalysisTransform analysis;
        private int assignmentIndex;
        LivenessAnalysisTransform.Liveness[] liveness;

        internal DeadCode2Transform(ModelCompiler compiler)
        {
            this.compiler = compiler;
        }

        public override ITypeDeclaration Transform(ITypeDeclaration itd)
        {
            analysis = new LivenessAnalysisTransform(compiler);
            Stopwatch watch = null;
            if (compiler.ShowProgress)
            {
                Console.Write($"({analysis.Name} ");
                watch = Stopwatch.StartNew();
            }
            analysis.Context.InputAttributes = context.InputAttributes;
            analysis.Transform(itd);
            if (compiler.ShowProgress)
            {
                watch.Stop();
                Console.Write("{0}ms) ", watch.ElapsedMilliseconds);
            }
            context.Results = analysis.Context.Results;
            if (!context.Results.IsSuccess)
            {
                Error("analysis failed");
                return itd;
            }
            var itdOut = base.Transform(itd);
            return itdOut;
        }

        protected override IFieldDeclaration ConvertField(ITypeDeclaration td, IFieldDeclaration ifd)
        {
            if (!analysis.usedFields.Contains(ifd))
            {
                if (debug)
                    Trace.WriteLine($"removing field {ifd}");
                return null;
            }
            return base.ConvertField(td, ifd);
        }

        protected override void DoConvertMethodBody(IList<IStatement> outputs, IList<IStatement> inputs)
        {
            // get the dead flags for this method
            IMethodDeclaration imd = context.FindAncestor<IMethodDeclaration>();
            int methodIndex = analysis.analysis.IndexOfMethod[imd];
            this.liveness = analysis.LivenessInMethod[methodIndex];
            // as we convert assignments, increment a counter
            // if isDead array says dead, return null
            assignmentIndex = 0;
            base.DoConvertMethodBody(outputs, inputs);
        }

        protected override IExpression ConvertAssign(IAssignExpression iae)
        {
            int index = assignmentIndex;
            assignmentIndex++;
            var ae = (IAssignExpression)base.ConvertAssign(iae);
            if (ae != null)
            {
                var lhsLiveness = liveness[index];
                if (lhsLiveness == LivenessAnalysisTransform.Liveness.Unused)
                    return null;
                else if (lhsLiveness == LivenessAnalysisTransform.Liveness.Dead)
                {
                    //Trace.WriteLine($"dead assign: {iae}");
                    if (ae.Target is IVariableDeclarationExpression)
                    {
                        if (ae.Expression is IDefaultExpression)
                            return ae;
                        else
                            return Builder.AssignExpr(ae.Target, Builder.DefaultExpr(ae.Target.GetExpressionType()));
                    }
                    else
                        return null;
                }
            }
            return ae;
        }
    }

    internal class LivenessAnalysisTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get
            {
                return "LivenessAnalysisTransform";
            }
        }

        public static bool debug;
        readonly ModelCompiler compiler;
        public MethodAnalysisTransform analysis;
        public Dictionary<int, Liveness[]> LivenessInMethod = new Dictionary<int, Liveness[]>();
        Set<IFieldDeclaration> liveFields = new Set<IFieldDeclaration>();
        public Set<IFieldDeclaration> usedFields = new Set<IFieldDeclaration>();
        Set<IFieldDeclaration> fieldsReadBeforeWrittenInPublicMethods = new Set<IFieldDeclaration>();
        Set<IVariableDeclaration> liveVariables = new Set<IVariableDeclaration>();
        Set<IVariableDeclaration> usedVariables = new Set<IVariableDeclaration>();
        Liveness[] LivenessThisMethod;
        int assignmentIndex;
        int currentMethodIndex;
        Set<int> methodsToAnalyze = new Set<int>();
        MethodDetails[] detailsOfMethod;
        Set<int> publicMethods = new Set<int>();
        int methodsAnalyzedCount;

        public class MethodDetails
        {
            public Set<IFieldDeclaration> liveFieldsAtEndOfMethod = new Set<IFieldDeclaration>();
            /// <summary>
            /// This is an over-estimate.
            /// </summary>
            public Set<IFieldDeclaration> fieldsReadBeforeWritten;
            /// <summary>
            /// This is an under-estimate.
            /// </summary>
            public Set<IFieldDeclaration> fieldsWrittenBeforeRead;
        }

        /// <summary>
        /// Labels assignments.
        /// </summary>
        public enum Liveness
        {
            /// <summary>
            /// The value of the assignment is used.
            /// </summary>
            Live,
            /// <summary>
            /// The value of the assignment is overwritten before being used.
            /// </summary>
            Dead,
            /// <summary>
            /// The variable on the left-hand side is never used.
            /// </summary>
            Unused
        }

        internal LivenessAnalysisTransform(ModelCompiler compiler)
        {
            this.compiler = compiler;
        }

        public override ITypeDeclaration Transform(ITypeDeclaration itd)
        {
            analysis = new MethodAnalysisTransform();
            Stopwatch watch = null;
            if (compiler.ShowProgress)
            {
                Console.Write($"({analysis.Name} ");
                watch = Stopwatch.StartNew();
            }
            analysis.Context.InputAttributes = context.InputAttributes;
            analysis.Transform(itd);
            if (compiler.ShowProgress)
            {
                watch.Stop();
                Console.Write("{0}ms) ", watch.ElapsedMilliseconds);
            }
            context.Results = analysis.Context.Results;
            if (!context.Results.IsSuccess)
            {
                Error("analysis failed");
                return itd;
            }
            if (debug)
            {
                Trace.WriteLine("method graph:");
                Trace.WriteLine(analysis.MethodGraph);
            }
            Initialise();
            CheckAcyclic(analysis.MethodGraph);
            // collect the set of public methods
            foreach(var methodIndex in analysis.MethodGraph.Nodes)
            {
                if(analysis.methods[methodIndex].Declaration.Visibility.HasFlag(MethodVisibility.Public))
                {
                    publicMethods.Add(methodIndex);
                }
            }
            detailsOfMethod = Util.ArrayInit(analysis.MethodGraph.Nodes.Count, methodIndex => new MethodDetails());
            // runs faster with CallersBeforeCallees
            var methodOrder = GetCallersBeforeCallees(analysis.MethodGraph);
            methodsToAnalyze.AddRange(methodOrder);
            while (methodsToAnalyze.Count > 0)
            {
                foreach (int methodIndex in methodOrder)
                {
                    if (methodsToAnalyze.Contains(methodIndex))
                    {
                        // AnalyzeMethod may add new methods to methodsToAnalyze.
                        AnalyzeMethod(methodIndex);
                    }
                }
            }
            if (debug)
                Trace.WriteLine($"{this.Name}: {analysis.MethodGraph.Nodes.Count} methods, {methodsAnalyzedCount} methods analyzed");
            return itd;
        }

        private List<int> GetCalleesBeforeCallers(IndexedGraph g)
        {
            List<int> calleesBeforeCallers = new List<int>();
            DepthFirstSearch<int> dfs = new DepthFirstSearch<int>(g);
            dfs.FinishNode += calleesBeforeCallers.Add;
            dfs.SearchFrom(g.Nodes);
            return calleesBeforeCallers;
        }

        private List<int> GetCallersBeforeCallees(IndexedGraph g)
        {
            List<int> callersBeforeCallees = new List<int>();
            DepthFirstSearch<int> dfs = new DepthFirstSearch<int>(g.SourcesOf, g);
            dfs.FinishNode += callersBeforeCallees.Add;
            dfs.SearchFrom(g.Nodes);
            return callersBeforeCallees;
        }

        protected void AnalyzeMethod(int methodIndex)
        {
            methodsAnalyzedCount++;
            //Trace.WriteLine($"AnalyzeMethod {methodIndex} {analysis.methods[methodIndex].Declaration.Name}");
            int oldMethodIndex = currentMethodIndex;
            currentMethodIndex = methodIndex;
            var methodDetails = detailsOfMethod[methodIndex];
            liveFields.Clear();
            liveFields.AddRange(methodDetails.liveFieldsAtEndOfMethod);
            ConvertMethod(analysis.methods[methodIndex].Declaration);
            methodsToAnalyze.Remove(methodIndex);
            bool changed = false;
            var fieldsWrittenBeforeRead = methodDetails.liveFieldsAtEndOfMethod - liveFields;
            if (methodDetails.fieldsWrittenBeforeRead == null || fieldsWrittenBeforeRead != methodDetails.fieldsWrittenBeforeRead)
            {
                methodDetails.fieldsWrittenBeforeRead = fieldsWrittenBeforeRead;
                changed = true;
            }
            var fieldsReadBeforeWritten = liveFields - methodDetails.liveFieldsAtEndOfMethod;
            if (methodDetails.fieldsReadBeforeWritten == null || fieldsReadBeforeWritten != methodDetails.fieldsReadBeforeWritten)
            {
                methodDetails.fieldsReadBeforeWritten = fieldsReadBeforeWritten;
                changed = true;
            }
            if (changed)
            {
                foreach (int caller in analysis.MethodGraph.SourcesOf(currentMethodIndex))
                {
                    methodsToAnalyze.Add(caller);
                }
            }
            if (publicMethods.Contains(methodIndex))
            {
                // if the method is public, add fieldsReadBeforeWritten to fieldsReadBeforeWrittenInPublicMethods
                // if changed, must add to liveFieldsAtEndOfMethod and re-analyze all public methods
                if(!fieldsReadBeforeWrittenInPublicMethods.ContainsAll(fieldsReadBeforeWritten))
                {
                    fieldsReadBeforeWrittenInPublicMethods.AddRange(fieldsReadBeforeWritten);
                    foreach(var publicMethodIndex in publicMethods)
                    {
                        // note publicMethodIndex can equal methodIndex
                        var publicMethodDetails = detailsOfMethod[publicMethodIndex];
                        if (!publicMethodDetails.liveFieldsAtEndOfMethod.ContainsAll(fieldsReadBeforeWritten))
                        {
                            publicMethodDetails.liveFieldsAtEndOfMethod.AddRange(fieldsReadBeforeWritten);
                            methodsToAnalyze.Add(publicMethodIndex);
                        }
                    }
                }
            }
            currentMethodIndex = oldMethodIndex;
        }

        protected void CheckAcyclic(IndexedGraph g)
        {
            DepthFirstSearch<int> dfs = new DepthFirstSearch<int>(g);
            dfs.BackEdge += delegate (Edge<int> edge)
            {
                throw new Exception("back edge");
            };
            dfs.SearchFrom(g.Nodes);
        }

        protected override IStatement ConvertReturnStatement(IMethodReturnStatement imrs)
        {
            // include fields that were live at the end of the method.
            liveFields.AddRange(detailsOfMethod[currentMethodIndex].liveFieldsAtEndOfMethod);
            return base.ConvertReturnStatement(imrs);
        }

        /// <summary>
        /// Analyze in reverse order.
        /// </summary>
        /// <param name="ics"></param>
        /// <returns></returns>
        protected override IStatement ConvertCondition(IConditionStatement ics)
        {
            if (ics.Else != null)
                throw new Exception("ics.Else != null");
            var liveFieldsAtEnd = (Set<IFieldDeclaration>)liveFields.Clone();
            var liveVariablesAtEnd = (Set<IVariableDeclaration>)liveVariables.Clone();
            ConvertBlock(ics.Then);
            ConvertExpression(ics.Condition);
            // merge the live variables at end since the condition may not be taken.
            liveFields.AddRange(liveFieldsAtEnd);
            liveVariables.AddRange(liveVariablesAtEnd);
            return ics;
        }

        protected override IStatement ConvertFor(IForStatement ifs)
        {
            // We process the loop body twice to catch cyclic dependencies.
            // A loop is expanded into the following:
            // Initializer
            // Condition
            // Body
            // Increment
            // Condition
            // Body
            // Increment
            // Condition
            var liveFieldsAtEnd = (Set<IFieldDeclaration>)liveFields.Clone();
            var liveVariablesAtEnd = (Set<IVariableDeclaration>)liveVariables.Clone();
            var oldAssignmentIndex = assignmentIndex;
            var wasDeadThisMethod = LivenessThisMethod;
            // The first time through the body, we collect liveness but don't mark assignments as dead.
            LivenessThisMethod = null;
            ConvertExpression(ifs.Condition);
            ConvertStatement(ifs.Increment);
            ConvertBlock(ifs.Body);
            LivenessThisMethod = wasDeadThisMethod;
            assignmentIndex = oldAssignmentIndex;
            // merge the live variables at end since the loop may iterate only once.
            liveFields.AddRange(liveFieldsAtEnd);
            liveVariables.AddRange(liveVariablesAtEnd);
            ConvertExpression(ifs.Condition);
            ConvertStatement(ifs.Increment);
            ConvertBlock(ifs.Body);
            // merge the live variables at end since the loop may not be entered.
            liveFields.AddRange(liveFieldsAtEnd);
            liveVariables.AddRange(liveVariablesAtEnd);
            ConvertExpression(ifs.Condition);
            ConvertStatement(ifs.Initializer);
            return ifs;
        }

        /// <summary>
        /// Analyzes statements in reverse order.
        /// </summary>
        /// <param name="outputs"></param>
        /// <param name="inputs"></param>
        protected override void ConvertStatements(IList<IStatement> outputs, IEnumerable<IStatement> inputs)
        {
            OpenOutputBlock(outputs);
            List<IStatement> inputList = inputs.ToList();
            for (int i = inputList.Count - 1; i >= 0; i--)
            {
                IStatement ist = inputList[i];
                ConvertStatement(ist);
            }
            outputs.AddRange(inputs);
            CloseOutputBlock();
        }

        protected override IMethodDeclaration DoConvertMethod(IMethodDeclaration md, IMethodDeclaration imd)
        {
            List<IStatement> outputs = new List<IStatement>();
            DoConvertMethodBody(outputs, imd.Body.Statements);
            return imd;
        }

        protected override void DoConvertMethodBody(IList<IStatement> outputs, IList<IStatement> inputs)
        {
            // change globals
            liveVariables.Clear();
            usedVariables.Clear();
            var currentMethod = analysis.methods[currentMethodIndex];
            var wasDeadThisMethod = LivenessThisMethod;
            LivenessThisMethod = new Liveness[currentMethod.AssignmentCount];
            LivenessInMethod[currentMethodIndex] = LivenessThisMethod;
            int oldAssignmentIndex = assignmentIndex;
            assignmentIndex = currentMethod.AssignmentCount;

            // do the analysis
            base.DoConvertMethodBody(outputs, inputs);
            if (assignmentIndex != 0)
                throw new Exception("assignmentIndex != 0");
            liveVariables.Remove(analysis.loopVariables);
            if (liveVariables.Count != 0)
                Error($"liveVariables.Count != 0 ({liveVariables})");
            if (debug && false)
            {
                var imd = currentMethod.Declaration;
                Trace.WriteLine(imd.Name);
                Trace.WriteLine("live variables:");
                Trace.WriteLine(liveVariables);
                Trace.WriteLine("live fields:");
                Trace.WriteLine(liveFields);
            }

            // restore globals
            this.LivenessThisMethod = wasDeadThisMethod;
            this.assignmentIndex = oldAssignmentIndex;
        }

        protected override IExpression ConvertAssign(IAssignExpression iae)
        {
            assignmentIndex--;
            //Trace.WriteLine($"{analysis.methods[currentMethodIndex].Declaration.Name} {assignmentIndex} {iae}");
            bool assignmentIsDead = false;
            // do not convert the target, only its indices
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(iae.Target);
            if (ivd != null)
            {
                if (!liveVariables.Contains(ivd) && !analysis.loopVariables.Contains(ivd))
                    assignmentIsDead = true;
            }
            IFieldReference ifr = Recognizer.GetFieldReference(iae.Target);
            IFieldDeclaration ifd = ifr?.Resolve();
            if (ifd != null)
            {
                if (!liveFields.Contains(ifd))
                    assignmentIsDead = true;
            }
            if (iae.Target is IVariableReferenceExpression || iae.Target is IVariableDeclarationExpression)
            {
                liveVariables.Remove(ivd);
            }
            else if (iae.Target is IFieldReferenceExpression ifre && ifre.Field.Equals(ifr))
            {
                liveFields.Remove(ifd);
            }
            if (!assignmentIsDead)
            {
                if (ivd != null)
                    usedVariables.Add(ivd);
                else if (ifd != null)
                    usedFields.Add(ifd);
                // analyze the indices of target
                foreach(var bracket in Recognizer.GetIndices(iae.Target))
                    ConvertCollection(bracket);
                // must convert the expression afterward, since it may use target.
                base.ConvertExpression(iae.Expression);
            }
            else if (LivenessThisMethod != null)
            {
                bool variableIsUsed = (ivd != null && usedVariables.Contains(ivd)) ||
                    (ifd != null && usedFields.Contains(ifd));
                LivenessThisMethod[assignmentIndex] = variableIsUsed ? Liveness.Dead : Liveness.Unused;
                if (debug && false)
                {
                    Trace.WriteLine($"dead statement: {iae}");
                    if (!variableIsUsed)
                        Trace.WriteLine($"unused variable: {iae.Target}");
                }
            }
            return iae;
        }

        protected override IExpression ConvertMethodInvoke(IMethodInvokeExpression imie)
        {
            int methodIndex;
            if (analysis.IndexOfMethod.TryGetValue(imie.Method.Method, out methodIndex))
            {
                MethodDetails methodDetails = detailsOfMethod[methodIndex];
                if (!methodDetails.liveFieldsAtEndOfMethod.ContainsAll(liveFields))
                {
                    methodDetails.liveFieldsAtEndOfMethod.AddRange(liveFields);
                    methodsToAnalyze.Add(methodIndex);
                }
                if(methodsToAnalyze.Contains(methodIndex))
                {
                    var liveFieldsAfterInvoke = (Set<IFieldDeclaration>)liveFields.Clone();
                    var liveVariablesAfterInvoke = (Set<IVariableDeclaration>)liveVariables.Clone();
                    var usedVariablesAfterInvoke = (Set<IVariableDeclaration>)usedVariables.Clone();
                    AnalyzeMethod(methodIndex);
                    liveFields = liveFieldsAfterInvoke;
                    liveVariables = liveVariablesAfterInvoke;
                    usedVariables = usedVariablesAfterInvoke;
                }
                liveFields.AddRange(methodDetails.fieldsReadBeforeWritten);
                liveFields.Remove(methodDetails.fieldsWrittenBeforeRead);
            }
            return base.ConvertMethodInvoke(imie);
        }

        protected override IExpression ConvertFieldRefExpr(IFieldReferenceExpression ifre)
        {
            IFieldDeclaration ifd = ifre.Field.Resolve();
            if (ifd != null)
            {
                liveFields.Add(ifd);
                usedFields.Add(ifd);
            }
            return base.ConvertFieldRefExpr(ifre);
        }

        protected override IExpression ConvertVariableRefExpr(IVariableReferenceExpression ivre)
        {
            liveVariables.Add(ivre.Variable.Variable);
            return base.ConvertVariableRefExpr(ivre);
        }
    }

    /// <summary>
    /// Constructs a call graph and collects the set of fields used in public methods.
    /// </summary>
    internal class MethodAnalysisTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get
            {
                return "MethodAnalysisTransform";
            }
        }

        public class MethodInfo
        {
            public IMethodDeclaration Declaration;
            public int AssignmentCount;

            public MethodInfo(IMethodDeclaration imd)
            {
                Declaration = imd;
            }
        }

        public List<MethodInfo> methods = new List<MethodInfo>();
        public Dictionary<IMethodReference, int> IndexOfMethod = new Dictionary<IMethodReference, int>();
        public IndexedGraph MethodGraph = new IndexedGraph();
        public Set<IVariableDeclaration> loopVariables = new Set<IVariableDeclaration>();
        private MethodInfo currentMethod;

        protected override IMethodDeclaration ConvertMethod(IMethodDeclaration imd)
        {
            int methodIndex = GetMethodIndex(imd);
            currentMethod = methods[methodIndex];
            // reset the declaration since it may not exactly match the references in the code.
            currentMethod.Declaration = imd;
            return base.ConvertMethod(imd);
        }

        protected override IExpression ConvertMethodInvoke(IMethodInvokeExpression imie)
        {
            if (imie.Method.Target is IThisReferenceExpression)
            {
                IMethodDeclaration imd = context.FindAncestor<IMethodDeclaration>();
                int sourceMethod = GetMethodIndex(imd);
                int targetMethod = GetMethodIndex(imie.Method.Method);
                MethodGraph.AddEdge(sourceMethod, targetMethod);
                //Trace.WriteLine($"{sourceMethod} {methods[sourceMethod].Name} called {targetMethod} {methods[targetMethod].Name}");
            }
            return base.ConvertMethodInvoke(imie);
        }

        private int GetMethodIndex(IMethodReference imr)
        {
            int index;
            if (!IndexOfMethod.TryGetValue(imr, out index))
            {
                index = IndexOfMethod.Count;
                IndexOfMethod.Add(imr, index);
                methods.Add(new MethodInfo(imr.Resolve()));
                MethodGraph.AddNode();
            }
            return index;
        }

        protected override IStatement ConvertFor(IForStatement ifs)
        {
            loopVariables.Add(Recognizer.LoopVariable(ifs));
            return base.ConvertFor(ifs);
        }

        protected override IExpression ConvertAssign(IAssignExpression iae)
        {
            //Trace.WriteLine($"{currentMethod.Declaration.Name} {currentMethod.AssignmentCount} {iae}");
            currentMethod.AssignmentCount++;
            return base.ConvertAssign(iae);
        }
    }
}
