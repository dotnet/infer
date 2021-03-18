// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Compiler.Attributes;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
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
            get { return (decl != null) ? decl.VariableType.DotNetType : varInfo.VariableType.DotNetType; }
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

        public IExpression ReplaceWithUsesChannel(IExpression expr, IExpression usageIndex)
        {
            List<IList<IExpression>> indices = Recognizer.GetIndices(expr, out IExpression target);
            IExpression newExpr;
            if (target is IVariableDeclarationExpression) newExpr = (usageIndex == null) ? (IExpression)Builder.VarDeclExpr(decl) : Builder.VarRefExpr(decl);
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
            return $"ChannelInfo({decl.ToString()},isUse={IsUse},isMarginal={IsMarginal})";
        }
    }
}
