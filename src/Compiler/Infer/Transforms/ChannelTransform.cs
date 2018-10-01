// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Algorithms;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Models;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// THIS TRANSFORM IS DEPRECATED.  Use VariableTransform and Channel2Transform instead.
    /// Transforms variable references into channels, by duplicating variables.  
    /// A channel is a variable which is assigned once and referenced only once.  
    /// It corresponds to an edge in a factor graph.
    /// </summary>
    internal class ChannelTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "ChannelTransform"; }
        }

        private ChannelAnalysisTransform analysis;

        public override ITypeDeclaration Transform(ITypeDeclaration itd)
        {
            analysis = new ChannelAnalysisTransform();
            analysis.Context.InputAttributes = context.InputAttributes;
            analysis.Transform(itd);
            context.Results = analysis.Context.Results;
            return base.Transform(itd);
        }

        public IAlgorithm algorithm;

        public ChannelTransform(IAlgorithm algorithm)
        {
            this.algorithm = algorithm;
        }

        /// <summary>
        /// Only converts the contained statements in a for loop, leaving the initializer,
        /// condition and increment statements unchanged.
        /// </summary>
        /// <remarks>This method includes a number of checks to ensure the loop is valid e.g. that the initializer, condition and increment 
        /// are all of the appropriate form.</remarks>
        protected override IStatement ConvertFor(IForStatement ifs)
        {
            IForStatement fs = Builder.ForStmt();
            context.SetPrimaryOutput(fs);
            // Check condition is valid
            fs.Condition = ifs.Condition;
            if ((!(fs.Condition is IBinaryExpression)) || (((IBinaryExpression) fs.Condition).Operator != BinaryOperator.LessThan))
                Error("For statement condition must be of the form 'indexVar<loopSize', was " + fs.Condition);

            // Check increment is valid
            fs.Increment = ifs.Increment;
            IExpressionStatement ies = fs.Increment as IExpressionStatement;
            bool validIncrement = false;
            if (ies != null)
            {
                if (ies.Expression is IAssignExpression)
                {
                    IAssignExpression iae = (IAssignExpression) ies.Expression;
                    IBinaryExpression ibe = RemoveCast(iae.Expression) as IBinaryExpression;
                    validIncrement = (ibe != null) && (ibe.Operator == BinaryOperator.Add);
                }
                else if (ies.Expression is IUnaryExpression)
                {
                    IUnaryExpression iue = (IUnaryExpression) ies.Expression;
                    validIncrement = (iue.Operator == UnaryOperator.PostIncrement);
                }
            }
            if (!validIncrement)
            {
                Error("For statement increment must be of the form 'varname++' or 'varname=varname+1', was " + fs.Increment + ".");
            }


            // Check initializer is valid
            fs.Initializer = ifs.Initializer;
            ies = fs.Initializer as IExpressionStatement;
            if (ies == null)
            {
                Error("For statement initializer must be an expression statement, was " + fs.Initializer.GetType());
            }
            else
            {
                if (!(ies.Expression is IAssignExpression))
                {
                    Error("For statement initializer must be an assignment, was " + fs.Initializer.GetType().Name);
                }
                else
                {
                    IAssignExpression iae2 = (IAssignExpression) ies.Expression;
                    if (!(iae2.Target is IVariableDeclarationExpression)) Error("For statement initializer must be a variable declaration, was " + iae2.Target.GetType().Name);
                    if (!Recognizer.IsLiteral(iae2.Expression, 0)) Error("Loop index must start at 0, was " + iae2.Expression);
                }
            }

            fs.Body = ConvertBlock(ifs.Body);
            return fs;
        }

        internal static IExpression RemoveCast(IExpression expr)
        {
            // used to remove spurious casts
            if (expr is ICastExpression) return ((ICastExpression) expr).Expression;
            return expr;
        }

        protected override IStatement DoConvertStatement(IStatement ist)
        {
            if ((ist is IForStatement) || (ist is IExpressionStatement) || (ist is IBlockStatement) || (ist is IConditionStatement))
            {
                return base.DoConvertStatement(ist);
            }
            Error("Unsupported statement type: " + ist.GetType().Name);
            return ist;
        }

        protected override IExpression ConvertAssign(IAssignExpression iae)
        {
            IExpression targ = Recognizer.StripIndexers(iae.Target);
            if (targ is IArgumentReferenceExpression)
            {
                IArgumentReferenceExpression iare = (IArgumentReferenceExpression) targ;
                if (!(iae.Expression is IMethodInvokeExpression)) Error("Cannot redefine the value of parameter '" + iare.Parameter.Name + "'.");
            }
            IAssignExpression ae;
            if (iae.Expression is IArrayCreateExpression)
            {
                // We need to process the target first, rather than last, as normal
                ae = Builder.AssignExpr();
                ae.Target = ConvertExpression(iae.Target);
                ae.Expression = ConvertExpression(iae.Expression);
                context.InputAttributes.CopyObjectAttributesTo(iae, context.OutputAttributes, ae);
            }
            else
            {
                ae = (IAssignExpression) base.ConvertAssign(iae);
            }
            if (ae.Target == null) return null;
            return ae;
        }

        public static void setAllGroupRoots(BasicTransformContext context, IVariableDeclaration ivd, bool isRoot)
        {
            IList<GroupMember> gmas = context.OutputAttributes.GetAll<GroupMember>(ivd);
            if (gmas != null)
            {
                context.OutputAttributes.Remove<GroupMember>(ivd);
                foreach (GroupMember gma in gmas)
                {
                    GroupMember gmanew = new GroupMember(gma.Group, isRoot);
                    context.OutputAttributes.Add(ivd, gmanew);
                }
            }
        }

        protected override IParameterDeclaration ConvertMethodParameter(IParameterDeclaration ipd, int index)
        {
            ProcessConstant(ipd);
            return base.ConvertMethodParameter(ipd, index);
        }

        private void ProcessConstant(object decl)
        {
            if (context.InputAttributes.Has<IsInferred>(decl))
            {
                VariableInformation vi = VariableInformation.GetVariableInformation(context, decl);
                vi.DefineAllIndexVars(context);
                MarginalPrototype mpa = Context.InputAttributes.Get<MarginalPrototype>(decl);
                if (!vi.SetMarginalPrototypeFromAttribute(mpa, throwIfMissing: false))
                {
                    IExpression varRef = null;
                    if (decl is IVariableDeclaration) varRef = Builder.VarRefExpr((IVariableDeclaration) decl);
                    else if (decl is IParameterDeclaration) varRef = Builder.ParamRef((IParameterDeclaration) decl);
                    else throw new NotImplementedException();
                    vi.marginalPrototypeExpression = Builder.NewObject(typeof (PointMass<>).MakeGenericType(vi.varType), varRef);
                }
            }
        }

        /// <summary>
        /// Converts a variable declaration by creating definition, marginal and uses channel variables.
        /// </summary>
        protected override IExpression ConvertVariableDeclExpr(IVariableDeclarationExpression ivde)
        {
            IVariableDeclaration ivd = ivde.Variable;
            VariableInformation vi = VariableInformation.GetVariableInformation(context, ivd);

            // If the variable is deterministic, return
            if (!vi.IsStochastic)
            {
                ProcessConstant(ivd);
                context.OutputAttributes.Set(ivd, new DescriptionAttribute("The constant '" + ivd.Name + "'"));
                return ivde;
            }
            bool suppressVariableFactor = context.InputAttributes.Has<SuppressVariableFactor>(ivd);
            bool isDerived = context.InputAttributes.Has<DerivedVariable>(ivd);
            bool isConstant = false;
            bool isInferred = context.InputAttributes.Has<IsInferred>(ivd);
            int useCount;
            ChannelAnalysisTransform.UsageInfo info;
            if (!analysis.usageInfo.TryGetValue(ivd, out info)) useCount = 0;
            else useCount = info.NumberOfUsesOld;
            if (!(algorithm is Algorithms.GibbsSampling) && !isConstant && !suppressVariableFactor && (useCount == 1) && !isInferred && isDerived)
            {
                // this is optional
                suppressVariableFactor = true;
                context.InputAttributes.Set(ivd, new SuppressVariableFactor());
            }
            context.InputAttributes.Remove<LoopContext>(ivd);
            context.InputAttributes.Set(ivd, new LoopContext(context));

            // Create variable-to-channel information for the variable.
            VariableToChannelInformation vtc = new VariableToChannelInformation();
            vtc.shareAllUses = (useCount == 1);
            Context.InputAttributes.Set(ivd, vtc);

            // Ensure the marginal prototype is set.
            MarginalPrototype mpa = Context.InputAttributes.Get<MarginalPrototype>(ivd);
            try
            {
                vi.SetMarginalPrototypeFromAttribute(mpa);
            }
            catch (ArgumentException ex)
            {
                Error(ex.Message);
            }

            // Create the definition channel 
            vtc.defChannel = ChannelInfo.DefChannel(vi);
            vtc.defChannel.decl = ivd;
            // Always create a variable factor for a stochastic variable
            if (!isConstant && !suppressVariableFactor)
            {
                vi.DefineAllIndexVars(context);
                IList<IStatement> stmts = Builder.StmtCollection();

                // Create marginal channel
                vtc.marginalChannel = ChannelInfo.MarginalChannel(vi);
                vtc.marginalChannel.decl = vi.DeriveIndexedVariable(stmts, context, vi.Name + "_marginal");
                context.InputAttributes.CopyObjectAttributesTo<InitialiseTo>(vi.declaration, context.OutputAttributes, vtc.marginalChannel.decl);
                context.OutputAttributes.Set(vtc.marginalChannel.decl, vtc.marginalChannel);
                context.OutputAttributes.Set(vtc.marginalChannel.decl, new DescriptionAttribute("marginal of '" + ivd.Name + "'"));
                SetMarginalPrototype(vtc.marginalChannel.decl);
                if (algorithm is GibbsSampling && ((GibbsSampling) algorithm).UseSideChannels)
                {
                    Type marginalType = MessageTransform.GetDistributionType(vi.varType, vi.InnermostElementType,
                                                                             vi.marginalPrototypeExpression.GetExpressionType(), true);
                    Type domainType = ivd.VariableType.DotNetType;

                    vtc.samplesChannel = ChannelInfo.MarginalChannel(vi);
                    vtc.samplesChannel.decl = vi.DeriveIndexedVariable(stmts, context, vi.Name + "_samples");
                    context.OutputAttributes.Remove<InitialiseTo>(vtc.samplesChannel.decl);
                    context.OutputAttributes.Set(vtc.samplesChannel.decl, vtc.samplesChannel);
                    context.OutputAttributes.Set(vtc.samplesChannel.decl, new DescriptionAttribute("samples of '" + ivd.Name + "'"));
                    Type samplesType = typeof (List<>).MakeGenericType(domainType);
                    IExpression samples_mpe = Builder.NewObject(samplesType);
                    VariableInformation samples_vi = VariableInformation.GetVariableInformation(context, vtc.samplesChannel.decl);
                    samples_vi.marginalPrototypeExpression = samples_mpe;

                    vtc.conditionalsChannel = ChannelInfo.MarginalChannel(vi);
                    vtc.conditionalsChannel.decl = vi.DeriveIndexedVariable(stmts, context, vi.Name + "_conditionals");
                    context.OutputAttributes.Remove<InitialiseTo>(vtc.conditionalsChannel.decl);
                    context.OutputAttributes.Set(vtc.conditionalsChannel.decl, vtc.conditionalsChannel);
                    context.OutputAttributes.Set(vtc.conditionalsChannel.decl, new DescriptionAttribute("conditionals of '" + ivd.Name + "'"));
                    Type conditionalsType = typeof (List<>).MakeGenericType(marginalType);
                    IExpression conditionals_mpe = Builder.NewObject(conditionalsType);
                    VariableInformation conditionals_vi = VariableInformation.GetVariableInformation(context, vtc.conditionalsChannel.decl);
                    conditionals_vi.marginalPrototypeExpression = conditionals_mpe;
                }
                else
                {
                    vtc.samplesChannel = vtc.marginalChannel;
                    vtc.conditionalsChannel = vtc.marginalChannel;
                }

                // Create uses channel
                vtc.usageChannel = ChannelInfo.UseChannel(vi);
                vtc.usageChannel.decl = vi.DeriveArrayVariable(stmts, context, vi.Name + "_uses", Builder.LiteralExpr(useCount), Builder.VarDecl("_ind", typeof (int)), useLiteralIndices: true);
                context.InputAttributes.CopyObjectAttributesTo<InitialiseTo>(vi.declaration, context.OutputAttributes, vtc.usageChannel.decl);
                context.OutputAttributes.Set(vtc.usageChannel.decl, vtc.usageChannel);
                context.OutputAttributes.Set(vtc.usageChannel.decl, new DescriptionAttribute("uses of '" + ivd.Name + "'"));
                SetMarginalPrototype(vtc.usageChannel.decl);

                //setAllGroupRoots(context, ivd, false);

                context.AddStatementsBeforeCurrent(stmts);

                // Append usageDepth indices to def/marginal/use expressions
                IExpression defExpr = Builder.VarRefExpr(ivd);
                IExpression marginalExpr = Builder.VarRefExpr(vtc.marginalChannel.decl);
                IExpression usageExpr = Builder.VarRefExpr(vtc.usageChannel.decl);
                IExpression countExpr = Builder.LiteralExpr(useCount);

                // Add clone factor tying together all of the channels 
                IMethodInvokeExpression usesEqualDefExpression;
                Type[] genArgs = new Type[] {vi.varType};
                if (algorithm is GibbsSampling && ((GibbsSampling) algorithm).UseSideChannels)
                {
                    GibbsSampling gs = (GibbsSampling) algorithm;
                    IExpression burnInExpr = Builder.LiteralExpr(gs.BurnIn);
                    IExpression thinExpr = Builder.LiteralExpr(gs.Thin);
                    IExpression samplesExpr = Builder.VarRefExpr(vtc.samplesChannel.decl);
                    IExpression conditionalsExpr = Builder.VarRefExpr(vtc.conditionalsChannel.decl);
                    if (isDerived)
                    {
                        Delegate d = new FuncOut3<PlaceHolder, int, int, int, PlaceHolder, PlaceHolder, PlaceHolder, PlaceHolder[]>(Factor.ReplicateWithMarginalGibbs);
                        usesEqualDefExpression = Builder.StaticGenericMethod(d, genArgs, defExpr, countExpr, burnInExpr, thinExpr, marginalExpr, samplesExpr, conditionalsExpr);
                    }
                    else
                    {
                        Delegate d = new FuncOut3<PlaceHolder, int, int, int, PlaceHolder, PlaceHolder, PlaceHolder, PlaceHolder[]>(Factor.UsesEqualDefGibbs);
                        usesEqualDefExpression = Builder.StaticGenericMethod(d, genArgs, defExpr, countExpr, burnInExpr, thinExpr, marginalExpr, samplesExpr, conditionalsExpr);
                    }
                }
                else
                {
                    Delegate d;
                    if (isDerived)
                    {
                        d = new FuncOut<PlaceHolder, int, PlaceHolder, PlaceHolder[]>(Factor.ReplicateWithMarginal<PlaceHolder>);
                    }
                    else
                    {
                        d = new FuncOut<PlaceHolder, int, PlaceHolder, PlaceHolder[]>(Factor.UsesEqualDef<PlaceHolder>);
                    }
                    usesEqualDefExpression = Builder.StaticGenericMethod(d, genArgs, defExpr, countExpr, marginalExpr);
                }
                if (isDerived) context.OutputAttributes.Set(usesEqualDefExpression, new DerivedVariable()); // used by Gibbs
                // Mark this as a pseudo-factor
                context.OutputAttributes.Set(usesEqualDefExpression, new IsVariableFactor());
                if (useCount == 1)
                    context.OutputAttributes.Set(usesEqualDefExpression, new DivideMessages(false));
                else
                    context.InputAttributes.CopyObjectAttributesTo<DivideMessages>(ivd, context.OutputAttributes, usesEqualDefExpression);
                context.InputAttributes.CopyObjectAttributesTo<GivePriorityTo>(ivd, context.OutputAttributes, usesEqualDefExpression);
                IAssignExpression assignExpr = Builder.AssignExpr(usageExpr, usesEqualDefExpression);

                // Copy attributes across from input to output
                Context.InputAttributes.CopyObjectAttributesTo<Algorithm>(ivd, context.OutputAttributes, assignExpr);
                context.OutputAttributes.Remove<InitialiseTo>(ivd);
                if (vi.ArrayDepth == 0)
                {
                    // Insert the UsesEqualDef statement after the declaration.
                    // Note the variable will not have been defined yet.
                    context.AddStatementAfterCurrent(Builder.ExprStatement(assignExpr));
                }
                else
                {
                    // For an array, the UsesEqualDef statement should be inserted after the array is allocated.
                    // Store the statement for later use by ConvertArrayCreate.
                    context.InputAttributes.Remove<LoopContext>(ivd);
                    context.InputAttributes.Set(ivd, new LoopContext(context));
                    context.InputAttributes.Remove<Containers>(ivd);
                    context.InputAttributes.Set(ivd, new Containers(context));
                    vtc.usesEqualDefsStatements = Builder.StmtCollection();
                    vtc.usesEqualDefsStatements.Add(Builder.ExprStatement(assignExpr));
                }
            }
            // These must be set after the above or they will be copied to the other channels
            context.OutputAttributes.Set(ivd, vtc.defChannel);
            context.OutputAttributes.Set(vtc.defChannel.decl, new DescriptionAttribute("definition of '" + ivd.Name + "'"));
            return ivde;
        }

        private void SetMarginalPrototype(IVariableDeclaration ivd)
        {
            VariableInformation vi = VariableInformation.GetVariableInformation(context, ivd);
            // Ensure the marginal prototype is set.
            MarginalPrototype mpa = Context.InputAttributes.Get<MarginalPrototype>(ivd);
            try
            {
                vi.SetMarginalPrototypeFromAttribute(mpa);
            }
            catch (ArgumentException ex)
            {
                Error(ex.Message);
            }
        }

        /// <summary>
        /// When array creations are assigned to stochastic arrays, this creates corresponding arrays for the marginal and uses channels.
        /// </summary>
        /// <param name="iace"></param>
        /// <returns></returns>
        protected override IExpression ConvertArrayCreate(IArrayCreateExpression iace)
        {
            IAssignExpression iae = context.FindAncestor<IAssignExpression>();
            if (iae == null) return iace;
            if (iae.Expression != iace) return iace;
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(iae.Target);
            VariableToChannelInformation vtci = context.InputAttributes.Get<VariableToChannelInformation>(ivd);
            if (vtci == null) return iace; // not a stochastic variable
            // Check if this is the last level of indexing
            bool lastLevel = (!(iace.Type is IArrayType));
            if ((lastLevel) && (vtci.usesEqualDefsStatements != null))
            {
                if (vtci.IsUsesEqualDefsStatementInserted)
                {
                    //Error("Duplicate array allocation.");
                }
                else
                {
                    // Insert the UsesEqualDef statement after the array is fully allocated.
                    // Note the array elements will not have been defined yet.
                    LoopContext lc = context.InputAttributes.Get<LoopContext>(ivd);
                    RefLoopContext rlc = lc.GetReferenceLoopContext(context);
                    // IMPORTANT TODO: add this statement at the right level!
                    IStatement ist = context.FindAncestor<IStatement>();
                    if (rlc.loops.Count > 0) ist = rlc.loops[0];
                    int ancIndex = context.GetAncestorIndex(ist);
                    Containers containers = context.InputAttributes.Get<Containers>(ivd);
                    Containers containersNeeded = containers.GetContainersNotInContext(context, ancIndex);
                    vtci.usesEqualDefsStatements = Containers.WrapWithContainers(vtci.usesEqualDefsStatements, containersNeeded.outputs);
                    context.AddStatementsAfter(ist, vtci.usesEqualDefsStatements);
                    vtci.IsUsesEqualDefsStatementInserted = true;
                }
            }
            return iace;
        }

        protected override IExpression ConvertMethodInvoke(IMethodInvokeExpression imie)
        {
            if (CodeRecognizer.IsInfer(imie)) return ConvertInfer(imie);
            return base.ConvertMethodInvoke(imie);
        }

        /// <summary>
        /// Modify the argument of Infer to be the marginal channel variable i.e. Infer(a) transforms to Infer(a_marginal).  
        /// </summary>
        /// <param name="imie"></param>
        /// <returns></returns>
        protected IExpression ConvertInfer(IMethodInvokeExpression imie)
        {
            IVariableReferenceExpression ivre = imie.Arguments[0] as IVariableReferenceExpression;
            if (ivre == null)
            {
                //Error("Argument to Infer() must be a variable reference, was " + imie.Arguments[0] + ".");
                return imie;
            }
            // Find expression for the marginal of interest
            IVariableDeclaration ivd = ivre.Variable.Resolve();
            VariableToChannelInformation vtci = context.InputAttributes.Get<VariableToChannelInformation>(ivd);
            if (vtci == null) return imie; // The argument is constant
            ExpressionEvaluator eval = new ExpressionEvaluator();
            QueryType query = (QueryType) eval.Evaluate(imie.Arguments[2]);
            IVariableDeclaration inferDecl = null;
            if (query == QueryTypes.Marginal) inferDecl = vtci.marginalChannel.decl;
            else if (query == QueryTypes.Samples) inferDecl = vtci.samplesChannel.decl;
            else if (query == QueryTypes.Conditionals) inferDecl = vtci.conditionalsChannel.decl;
            else return imie; // Error("Unrecognized query '"+query+"'");
            IMethodInvokeExpression mie = Builder.MethodInvkExpr();
            mie.Method = imie.Method;
            mie.Arguments.Add(Builder.VarRefExpr(inferDecl));
            for (int i = 1; i < imie.Arguments.Count; i++)
            {
                mie.Arguments.Add(imie.Arguments[i]);
            }
            // move the IsInferred attribute to the inferred channel
            context.OutputAttributes.Remove<IsInferred>(ivd);
            if (!context.OutputAttributes.Has<IsInferred>(inferDecl))
            {
                context.OutputAttributes.Set(inferDecl, new IsInferred());
                context.OutputAttributes.Add(inferDecl, new QueryTypeCompilerAttribute(query));
            }
            return mie;
        }

        /// <summary>
        /// Converts an array indexed expression.
        /// </summary>
        /// <param name="iaie"></param>
        /// <returns></returns>
        protected override IExpression ConvertArrayIndexer(IArrayIndexerExpression iaie)
        {
            IExpression expr = base.ConvertArrayIndexer(iaie);
            // Check if this is the top level indexer
            if (Recognizer.IsBeingIndexed(context)) return expr;
            IExpression target;
            List<IList<IExpression>> indices = Recognizer.GetIndices(expr, out target);
            if (!(target is IVariableReferenceExpression)) return expr;
            IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(target);
            if (ivd == null) return expr;
            VariableToChannelInformation vtci = Context.InputAttributes.Get<VariableToChannelInformation>(ivd);
            if (vtci == null) return expr;
            if (vtci.usageChannel == null) return expr;
            bool isDef = Recognizer.IsBeingMutated(context, expr);
            if (isDef) return expr;
            return vtci.usageChannel.ReplaceWithUsesChannel(expr, Builder.LiteralExpr(vtci.shareAllUses ? 0 : vtci.useCount++));
        }

        /// <summary>
        /// Converts a variable reference.
        /// </summary>
        /// <param name="ivre"></param>
        /// <returns></returns>
        protected override IExpression ConvertVariableRefExpr(IVariableReferenceExpression ivre)
        {
            IVariableDeclaration ivd = ivre.Variable.Resolve();
            VariableToChannelInformation vtci = Context.InputAttributes.Get<VariableToChannelInformation>(ivd);
            // If deterministic variable do nothing.
            if (vtci == null || vtci.usageChannel == null) return ivre;
            else if (Recognizer.IsBeingIndexed(context)) return ivre;
            else if (Recognizer.IsBeingMutated(context, ivre)) return ivre;
            else return vtci.usageChannel.ReplaceWithUsesChannel(ivre, Builder.LiteralExpr(vtci.shareAllUses ? 0 : vtci.useCount++));
        }

        /// <summary>
        /// Records information about the variable to channel transformation.
        /// </summary>
        private class VariableToChannelInformation : ICompilerAttribute
        {
            /*** Information associated with the channel transform ****/
            internal ChannelInfo samplesChannel, conditionalsChannel;
            internal ChannelInfo marginalChannel;
            internal ChannelInfo defChannel;
            internal ChannelInfo usageChannel;
            internal bool IsUsesEqualDefsStatementInserted = false;
            internal IList<IStatement> usesEqualDefsStatements;

            /// <summary>
            /// A running total of the number of uses transformed so far.
            /// </summary>
            internal int useCount;

            /// <summary>
            /// If true, all uses will share the same element of the usesArray.
            /// </summary>
            internal bool shareAllUses;
        }
    }

    /// <summary>
    /// Describes a channel, i.e. an edge in a factor graph.
    /// </summary>
    /// <remarks><para>
    /// A channel connects a model variable to a factor.
    /// A channel is either a definition, a use, or a replicated use of a model variable.
    /// </para><para>
    /// A definition channel connects a model variable to its unique defining factor.
    /// A definition channel appears in the inference program as a program variable having the same type as the model variable.
    /// </para><para>
    /// A use channel connects a model variable to a child factor.
    /// A use channel appears in the inference program as an array of the variable type.
    /// It is indexed via [usage indices][variable indices] where the variable indices are only needed
    /// if the variable is an array.
    /// </para>
    /// <para>
    /// All uses of a model variable are given the same ChannelInfo and packed into an array in the inference program.
    /// For example, if a model variable x has type <c>double</c>  and 2 uses, its usage channels will be declared together as <c>double[] x_uses = new double[3];</c> and referred to as x_uses[0] and x_uses[1].
    /// </para><para>
    /// The model variable type can be an array, in which case each channel is an array.
    /// For example, if a model variable x has type <c>double[]</c> and 2 uses, its usage channels will be declared together
    /// as <c>double[][] x_uses = new double[2][];</c>  and referred to as x_uses[0] and x_uses[1].
    /// </para><para>
    /// If the model variable is inside of a plate, then all instances of the channel share the same ChannelInfo and are
    /// packed into a nested array in the inference program.
    /// For example, if a model variable x is in a plate of size 3 and has 2 uses, its usage channels will be
    /// declared together as  <c>double[][] x_uses = new double[2][];</c> and x_uses[0] refers to the first use of x in the plate.
    /// </para></remarks>
    public class ChannelInfo : ICompilerAttribute
    {
        /// <summary>
        /// Helps build class declarations
        /// </summary>
        private static CodeBuilder Builder = CodeBuilder.Instance;

        /// <summary>
        /// Helps recognize code patterns
        /// </summary>
        private static CodeRecognizer Recognizer = CodeRecognizer.Instance;

        /// <summary>
        /// Information about the original variable from which the channels were created.
        /// </summary>
        internal readonly VariableInformation varInfo;

        /// <summary>
        /// Declaration of the channel.
        /// </summary>
        internal IVariableDeclaration decl;

        /// <summary>
        /// Flag to distinguish def/use channels.
        /// </summary>
        private readonly bool isUsage = false;

        /// <summary>
        /// Marks a channel as a marginal 
        /// </summary>
        public readonly bool IsMarginal;

        /// <summary>
        /// The type of the channel
        /// </summary>
        internal Type channelType
        {
            get { return decl.VariableType.DotNetType; }
        }

        /// <summary>
        /// True if the channel is a definition channel.
        /// </summary>
        public bool IsDef
        {
            get { return !IsMarginal && !isUsage; }
        }

        /// <summary>
        /// True if the channel is a usage channel.
        /// </summary>
        public bool IsUse
        {
            get { return isUsage; }
        }

        internal static ChannelInfo DefChannel(VariableInformation vi)
        {
            return new ChannelInfo(vi, false);
        }

        internal static ChannelInfo UseChannel(VariableInformation vi)
        {
            return new ChannelInfo(vi);
        }

        internal static ChannelInfo MarginalChannel(VariableInformation vi)
        {
            return new ChannelInfo(vi, true);
        }

        /// <summary>
        /// Creates information about a normal uses or defs channel.
        /// </summary>
        private ChannelInfo(VariableInformation vi)
        {
            this.varInfo = vi;
            this.isUsage = true;
            this.IsMarginal = false;
        }


        /// <summary>
        /// Creates information about a simple channel.
        /// There is no outer array.
        /// The inner array has size given by the varinfo.
        /// </summary>
        private ChannelInfo(VariableInformation vi, bool isMarginal)
        {
            this.varInfo = vi;
            this.IsMarginal = isMarginal;
        }

        ///// <summary>
        ///// Convert an array type into a distribution type.
        ///// </summary>
        ///// <param name="arrayType">A scalar, array, multidimensional array, or IList type.</param>
        ///// <param name="innermostElementType">Type of innermost array element (may be itself an array, if the array is compound).</param>
        ///// <param name="newInnermostElementType">Distribution type to use for the innermost array elements.</param>
        ///// <param name="useDistributionArrays">Convert outer arrays to DistributionArrays.</param>
        ///// <returns>A distribution type with the same structure as <paramref name="arrayType"/> but whose element type is <paramref name="newInnermostElementType"/>.</returns>
        ///// <remarks>
        ///// Similar to <see cref="Util.ChangeElementType"/> but converts arrays to DistributionArrays.
        ///// </remarks>
        //public static Type GetDistributionType(Type arrayType, Type innermostElementType, Type newInnermostElementType, bool useDistributionArrays)
        //{
        //   if (arrayType == innermostElementType) return newInnermostElementType;
        //   int rank;
        //   Type elementType = Util.GetElementType(arrayType, out rank);
        //   if (elementType == null) throw new ArgumentException(arrayType + " is not an array type with innermost element type " + innermostElementType);
        //   Type innerType = GetDistributionType(elementType, innermostElementType, newInnermostElementType, useDistributionArrays);
        //   if (useDistributionArrays)
        //   {
        //      return Distribution.MakeDistributionArrayType(innerType, rank);
        //   }
        //   else
        //   {
        //      return CodeBuilder.MakeArrayType(innerType, rank);
        //   }
        //}

        ///// <summary>
        ///// Convert an array type into a distribution type, converting a specified number of inner arrays to DistributionArrays.
        ///// </summary>
        ///// <param name="arrayType"></param>
        ///// <param name="innermostElementType"></param>
        ///// <param name="newInnermostElementType"></param>
        ///// <param name="depth">The current depth from the declaration type.</param>
        ///// <param name="useDistributionArraysDepth">The number of inner arrays to convert to DistributionArrays</param>
        ///// <returns></returns>
        //public static Type GetDistributionType(Type arrayType, Type innermostElementType, Type newInnermostElementType, int depth, int useDistributionArraysDepth)
        //{
        //   if (arrayType == innermostElementType) return newInnermostElementType;
        //   int rank;
        //   Type elementType = Util.GetElementType(arrayType, out rank);
        //   if (elementType == null) throw new ArgumentException(arrayType + " is not an array type.");
        //   Type innerType = GetDistributionType(elementType, innermostElementType, newInnermostElementType, depth + 1, useDistributionArraysDepth);
        //   if ((depth >= useDistributionArraysDepth) && (useDistributionArraysDepth >= 0))
        //   {
        //      return Distribution.MakeDistributionArrayType(innerType, rank);
        //   }
        //   else
        //   {
        //      return CodeBuilder.MakeArrayType(innerType, rank);
        //   }
        //}

        //public Type GetMessageType()
        //{
        //   int distArraysDepth = 0;
        //   if (IsUse) distArraysDepth = 1; // first array is [], then distribution arrays
        //   Type domainType = Distribution.GetDomainType(varInfo.marginalType);
        //   Type messageType = GetDistributionType(channelType, domainType, varInfo.marginalType, 0, distArraysDepth);
        //   return messageType;
        //}

        public IExpression ReplaceWithUsesChannel(IExpression expr, IExpression usageIndex)
        {
            IExpression target;
            List<IList<IExpression>> indices = Recognizer.GetIndices(expr, out target);
            IExpression newExpr = null;
            if (target is IVariableDeclarationExpression) newExpr = (usageIndex == null) ? (IExpression) Builder.VarDeclExpr(decl) : Builder.VarRefExpr(decl);
            else if (target is IVariableReferenceExpression) newExpr = Builder.VarRefExpr(decl);
            else throw new Exception("Unexpected indexing target: " + target);
            newExpr = Builder.ArrayIndex(newExpr, usageIndex);
            // append the remaining indices
            for (int i = 0; i < indices.Count; i++)
            {
                newExpr = Builder.ArrayIndex(newExpr, indices[i]);
            }
            return newExpr;
        }

        public override string ToString()
        {
            if (decl == null) return "ChannelInfo(decl=null)";
            return String.Format("ChannelInfo({0},isUse={1},isMarginal={2})", decl.ToString(), IsUse, IsMarginal);
        }
    }

    /// <summary>
    /// Attribute used to mark pseudo-factors corresponding to variables in the factor graph
    /// </summary>
    internal class IsVariableFactor : ICompilerAttribute
    {
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}