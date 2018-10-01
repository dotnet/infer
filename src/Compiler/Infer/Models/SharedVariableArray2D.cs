// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Models
{
    /// <summary>
    /// Interface for flat 2D shared variable arrays
    /// </summary>
    /// <typeparam name="DomainType">Domain type of the variable</typeparam>
    public interface SharedVariableArray2D<DomainType> : ISharedVariable
    {
        /// <summary>
        /// Gets the marginal
        /// </summary>
        /// <typeparam name="DistributionType">The returned distribution array type</typeparam>
        /// <returns></returns>
        DistributionType Marginal<DistributionType>();

        /// <summary>
        /// Get a copy of the variable array for the specified model
        /// </summary>
        /// <param name="model">The model id</param>
        /// <returns></returns>
        VariableArray2D<DomainType> GetCopyFor(Model model);

        /// <summary>
        /// Sets a copy of the shared variable to a definition
        /// </summary>
        /// <param name="model">Model id</param>
        /// <param name="definition">Defining variable</param>
        /// <returns></returns>
        void SetDefinitionTo(Model model, VariableArray2D<DomainType> definition);

        /// <summary>
        /// Inline method to name shared variable arrays
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        SharedVariableArray2D<DomainType> Named(string name);
    }

    /// <summary>
    /// A helper class that represents a variable array which is shared between multiple models.
    /// For example, where a very large model has been divided into sections corresponding to
    /// batches of data, an instance of this class can be used to help learn each parameter
    /// shared between the batches.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Shared variable arrays are used as follows. First the shared variable array is created with a prior distribution.
    /// Then a copy is created for each model using the <see cref="SharedVariableArray{DomainType}.GetCopyFor(Model)"/> method.
    /// Each model has a BatchCount which is the number of data batches you want to process with that model.
    /// Before performing inference in each model and batch, <see cref="SharedVariable{DomainType, DistributionType}.SetInput"/> should be called for each shared variable.
    /// After all shared variables have their inputs set, <see cref="SharedVariable{DomainType, DistributionType}.InferOutput(InferenceEngine,Model,int)"/> should then be called for each model and batch.
    /// These two steps are done automatically by <see cref="Model.InferShared(InferenceEngine,int)"/>.
    /// For inference to converge, you must loop multiple times through all the models, calling <see cref="Model.InferShared(InferenceEngine,int)"/> or SetInput/InferOutput each time.
    /// At any point the current marginal of the shared variable array can be retrieved using <see cref="SharedVariable{DomainType, DistributionType}.Marginal"/>.
    /// </para><para>In some situations, shared variable arrays cannot be created directly from a prior distribution, for
    /// example in a hierarchical model. In these situations, create the shared variable array with a uniform
    /// prior, and use <see cref="SharedVariableArray2D{DomainType}.SetDefinitionTo"/> to define the variable. 
    /// </para>
    /// </remarks>
    /// <typeparam name="DomainType">The domain type of an array element</typeparam>
    /// <typeparam name="DistributionArrayType">The marginal distribution type of the array</typeparam>
    internal class SharedVariableArray2D<DomainType, DistributionArrayType> : SharedVariable<DomainType[,], DistributionArrayType>, SharedVariableArray2D<DomainType>
        where DistributionArrayType : IDistribution<DomainType[,]>, Sampleable<DomainType[,]>, SettableToProduct<DistributionArrayType>, SettableToRatio<DistributionArrayType>
            ,
            ICloneable, SettableToUniform, SettableTo<DistributionArrayType>, CanGetLogAverageOf<DistributionArrayType>
    {
        /// <summary>
        /// Ranges for the array of shared variables
        /// </summary>
        public Range range0, range1;

        internal SharedVariableArray2D(Range range0, Range range1, DistributionArrayType prior, bool divideMessages = true)
            : base(prior, divideMessages)
        {
            this.range0 = range0;
            this.range1 = range1;
        }

        /// <summary>
        /// Inline method for naming an array of shared variables
        /// </summary>
        /// <param name="name">Name</param>
        /// <returns>this</returns>
        public new SharedVariableArray2D<DomainType, DistributionArrayType> Named(string name)
        {
            base.Named(name);
            return this;
        }

        SharedVariableArray2D<DomainType> SharedVariableArray2D<DomainType>.Named(string name)
        {
            return Named(name);
        }

        VariableArray2D<DomainType> SharedVariableArray2D<DomainType>.GetCopyFor(Model model)
        {
            if (model == DefiningModel)
                throw new ArgumentException("The shared variable is already defined by this model");

            Variable<DomainType[,]> v;
            if (!variables.TryGetValue(model, out v))
            {
                Variable<DistributionArrayType> vPrior = Variable.New<DistributionArrayType>()
                                                                 .Named(Name + "Prior");
                vPrior.ObservedValue = default(DistributionArrayType);
                VariableArray2D<DomainType> va = Variable.Array<DomainType>(range0, range1).Named(Name).Attrib(QueryTypes.MarginalDividedByPrior).Attrib(QueryTypes.Marginal);
                va.SetTo(Variable<DomainType[,]>.Random(vPrior));
                v = va;
                variables[model] = va;
                model.SharedVariables.Add(this);
                priors[model] = vPrior;
                DistributionArrayType[] messages = new DistributionArrayType[model.BatchCount];
                for (int i = 0; i < messages.Length; i++)
                {
                    messages[i] = (DistributionArrayType) Prior.Clone();
                    messages[i].SetToUniform();
                }
                if (DivideMessages)
                    CurrentMarginal = (DistributionArrayType) Prior.Clone();
                Outputs[model] = messages;
            }
            return (VariableArray2D<DomainType>) v;
        }

        void SharedVariableArray2D<DomainType>.SetDefinitionTo(Model model, VariableArray2D<DomainType> definition)
        {
            if (DefiningModel != null)
                throw new InvalidOperationException("Shared variable is already defined");

            if (model.BatchCount != 1)
                throw new ArgumentException("model.BatchCount != 1");

            if (!definition.IsBase)
                throw new ArgumentException("definition is a derived variable");

            Variable<DomainType[,]> v;
            if (!variables.TryGetValue(model, out v))
            {
                Variable<DistributionArrayType> vPrior = Variable.New<DistributionArrayType>()
                                                                 .Named(Name + "Constraint");
                vPrior.ObservedValue = default(DistributionArrayType);
                Variable.ConstrainEqualRandom<DomainType[,], DistributionArrayType>(definition, vPrior);
                variables[model] = definition;
                model.SharedVariables.Add(this);
                priors[model] = vPrior;
                DistributionArrayType[] messages = new DistributionArrayType[model.BatchCount];
                for (int i = 0; i < messages.Length; i++)
                {
                    messages[i] = (DistributionArrayType) Prior.Clone();
                    messages[i].SetToUniform();
                }
                CurrentMarginal = (DistributionArrayType) Prior.Clone();
                // In this case, output refers to the forward message from the definition.
                // There is only one as we are requiring that batch count = 1.
                Outputs[model] = messages;

                // This is the defining model for the variable
                DefiningModel = model;
            }
        }
    }
}