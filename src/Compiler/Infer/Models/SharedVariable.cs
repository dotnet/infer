// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Factors;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Math;
using Microsoft.ML.Probabilistic.Utilities;
using Microsoft.ML.Probabilistic.Models.Attributes;
using Microsoft.ML.Probabilistic.Compiler;

namespace Microsoft.ML.Probabilistic.Models
{
    /// <summary>
    /// Abstract base class for shared variables. Shared variables allow a model to be split
    /// into submodels in which variables are shared. Each submodel can have many copies. 
    /// </summary>
    /// <typeparam name="DomainType">Domian type of the variable</typeparam>
    /// <remarks>A typical use of this is for large data sets where the likelihood parts of the
    /// model cannot all fit in memory. The solution is to divide the data into chunks (or 'batches'), and specify
    /// a single submodel which includes the likelihood factors and variables for one chunk, along
    /// with the shared parameters; the number of copies of the submodel is set to the number
    /// of chunks. In a related pattern, there are one or more additional submodels for defining
    /// the parameter variables.</remarks>
    public abstract class SharedVariable<DomainType> : ISharedVariable
    {
        /// <summary>
        /// Name of the shared variable.
        /// </summary>
        public string Name;

        /// <summary>
        /// Creates a shared random variable with the specified prior distribution.
        /// </summary>
        /// <typeparam name="DistributionType">Distribution type</typeparam>
        /// <param name="prior">Prior</param>
        /// <param name="divideMessages">Use division (the faster default) for calculating messages to batches</param>
        /// <returns></returns>
        public static SharedVariable<DomainType> Random<DistributionType>(DistributionType prior, bool divideMessages = true)
            where DistributionType : IDistribution<DomainType>, Sampleable<DomainType>, SettableToProduct<DistributionType>,
                ICloneable, SettableToUniform, SettableTo<DistributionType>, SettableToRatio<DistributionType>, CanGetLogAverageOf<DistributionType>
        {
            return new SharedVariable<DomainType, DistributionType>(prior, divideMessages);
        }

        /// <summary>
        /// Creates a 1D array of shared random variables of size given by the specified range.
        /// </summary>
        /// <typeparam name="DistributionArrayType">The type of the supplied prior</typeparam>
        /// <param name="range">Range.</param>
        /// <param name="prior">A distribution over an array, to use as the prior.</param>
        /// <param name="divideMessages">Use division (the faster default) for calculating messages to batches</param>
        /// <returns></returns>
        public static SharedVariableArray<DomainType> Random<DistributionArrayType>(Range range, DistributionArrayType prior, bool divideMessages = true)
            where DistributionArrayType : IDistribution<DomainType[]>, Sampleable<DomainType[]>, SettableToProduct<DistributionArrayType>,
                ICloneable, SettableToUniform, SettableTo<DistributionArrayType>, SettableToRatio<DistributionArrayType>, CanGetLogAverageOf<DistributionArrayType>
        {
            return new SharedVariableArray<DomainType, DistributionArrayType>(range, prior, divideMessages);
        }

        /// <summary>
        /// Creates a 1D jagged array of shared random variables.
        /// </summary>
        /// <typeparam name="DistributionArrayType"></typeparam>
        /// <param name="itemPrototype">A fresh variable object representing an array element.</param>
        /// <param name="range">Outer range.</param>
        /// <param name="prior">Prior for the array.</param>
        /// <param name="divideMessages">Use division (the faster default) for calculating messages to batches</param>
        /// <returns></returns>
        public static ISharedVariableArray<VariableArray<DomainType>, DomainType[][]> Random<DistributionArrayType>(VariableArray<DomainType> itemPrototype, Range range,
                                                                                                                    DistributionArrayType prior, bool divideMessages = true)
            where DistributionArrayType : IDistribution<DomainType[][]>, Sampleable<DomainType[][]>, SettableToProduct<DistributionArrayType>,
                ICloneable, SettableToUniform, SettableTo<DistributionArrayType>, SettableToRatio<DistributionArrayType>, CanGetLogAverageOf<DistributionArrayType>
        {
            return new SharedVariableArray<VariableArray<DomainType>, DomainType[][], DistributionArrayType>(itemPrototype, range, prior, divideMessages);
        }

        /// <summary>
        /// Creates a generic jagged array of shared random variables.
        /// </summary>
        /// <typeparam name="ItemType">Item type</typeparam>
        /// <typeparam name="DistributionArrayType">Distribution array type</typeparam>
        /// <param name="itemPrototype">A fresh variable object representing an array element.</param>
        /// <param name="range">Outer range</param>
        /// <param name="prior">Prior for the array.</param>
        /// <param name="divideMessages">Use division (the faster default) for calculating messages to batches</param>
        /// <returns></returns>
        public static ISharedVariableArray<ItemType, DomainType> Random<ItemType, DistributionArrayType>(ItemType itemPrototype, Range range, DistributionArrayType prior,
                                                                                                         bool divideMessages = true)
            where ItemType : Variable, SettableTo<ItemType>, ICloneable
            where DistributionArrayType : IDistribution<DomainType>, Sampleable<DomainType>, SettableToProduct<DistributionArrayType>,
                ICloneable, SettableToUniform, SettableTo<DistributionArrayType>, SettableToRatio<DistributionArrayType>, CanGetLogAverageOf<DistributionArrayType>
        {
            return new SharedVariableArray<ItemType, DomainType, DistributionArrayType>(itemPrototype, range, prior, divideMessages);
        }

        /// <summary>
        /// Inline method for naming a shared variable.
        /// </summary>
        /// <param name="name">The name</param>
        /// <returns>this</returns>
        public SharedVariable<DomainType> Named(string name)
        {
            this.Name = name;
            return this;
        }

        /// <summary>
        /// ToString override.
        /// </summary>
        /// <returns></returns>
        /// <exclude/>
        public override string ToString()
        {
            return Name;
        }

        /// <summary>
        /// Get the marginal distribution for the shared variable, converted to type T.
        /// </summary>
        /// <typeparam name="T">The desired type</typeparam>
        /// <returns></returns>
        public abstract T Marginal<T>();

        /// <summary>
        /// Gets a copy of the variable for the specified model.
        /// </summary>
        /// <param name="model">Model id.</param>
        /// <returns></returns>
        public abstract Variable<DomainType> GetCopyFor(Model model);

        /// <summary>
        /// Sets the definition of the shared variable.
        /// </summary>
        /// <param name="model">Model id.</param>
        /// <param name="definition">Defining variable.</param>
        /// <returns></returns>
        /// <remarks>Use this method if the model is defining the shared variable rather than
        /// using one defined in this or another model.</remarks>
        public abstract void SetDefinitionTo(Model model, Variable<DomainType> definition);

        /// <summary>
        /// Sets the shared variable's inbox for a given model and batch.
        /// </summary>
        /// <param name="modelNumber"></param>
        /// <param name="batchNumber"></param>
        public abstract void SetInput(Model modelNumber, int batchNumber);

        /// <summary>
        /// Infer the shared variable's output message for the given model and batch number.
        /// </summary>
        /// <param name="engine">The inference engine.</param>
        /// <param name="modelNumber">The model id.</param>
        /// <param name="batchNumber">The batch number.</param>
        public abstract void InferOutput(InferenceEngine engine, Model modelNumber, int batchNumber);

        /// <summary>
        /// Infer the shared variable's output message for the given model and batch number.
        /// </summary>
        /// <param name="ca">The compiled algorithm.</param>
        /// <param name="modelNumber">The model id.</param>
        /// <param name="batchNumber">The batch number.</param>
        public abstract void InferOutput(IGeneratedAlgorithm ca, Model modelNumber, int batchNumber);

        /// <summary>
        /// Gets the evidence correction for this shared variable.
        /// </summary>
        /// <returns></returns>
        public abstract double GetEvidenceCorrection();

        /// <summary>
        /// Marks this shared variable as one that calculates evidence
        /// </summary>
        public bool IsEvidenceVariable { get; set; }
    }

#if false
    public interface SharedVariable<DomainType> : ISharedVariable
    {
        Variable<DomainType> GetCopyFor(Model model);
        SharedVariable<DomainType> Named(string name);
    }
#endif

    /// <summary>
    /// A helper class that represents a variable which is shared between multiple models.
    /// For example, where a very large model has been divided into sections corresponding to
    /// batches of data, an instance of this class can be used to help learn each parameter
    /// shared between the batches.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Shared variables are used as follows. First the shared variable is created with a prior distribution.
    /// Then a copy is created for each model using the <see cref="GetCopyFor"/> method.
    /// Each model has a BatchCount which is the number of data batches you want to process with that model.
    /// Before performing inference in each model and batch, <see cref="SetInput"/> should be called for each shared variable.
    /// After all shared variables have their inputs set, <see cref="InferOutput(InferenceEngine,Model,int)"/> should then be called for each model and batch.
    /// These two steps are done automatically by <see cref="Model.InferShared(InferenceEngine,int)"/>.
    /// For inference to converge, you must loop multiple times through all the models, calling <see cref="Model.InferShared(InferenceEngine,int)"/> or SetInput/InferOutput each time.
    /// At any point the current marginal of the shared variable can be retrieved using <see cref="Marginal"/>.
    /// </para>
    /// <para>In some situations, shared variables cannot be created directly from a prior distribution, for
    /// example in a hierarchical model. In these situations, create the shared variable with a uniform
    /// prior, and use <see cref="SetDefinitionTo"/> to define the variable. 
    /// </para>
    /// <para>A shared variable which calculates evidence must be treated as a special case; such variables can be marked
    /// using <see cref="SharedVariable{DomainType}.IsEvidenceVariable"/>, and the evidence is recovered using <see cref="Model.GetEvidenceForAll"/></para>
    /// </remarks>
    /// <typeparam name="DomainType">The domain type</typeparam>
    /// <typeparam name="DistributionType">The marginal distribution type</typeparam>
    internal class SharedVariable<DomainType, DistributionType> : SharedVariable<DomainType>
        where DistributionType : IDistribution<DomainType>, Sampleable<DomainType>, SettableToProduct<DistributionType>, SettableToRatio<DistributionType>,
            ICloneable, SettableToUniform, SettableTo<DistributionType>, CanGetLogAverageOf<DistributionType>
    {
        /// <summary>
        /// Prior
        /// </summary>
        protected DistributionType Prior;

        /// <summary>
        /// Marginal
        /// </summary>
        protected DistributionType CurrentMarginal;

        /// <summary>
        /// Dictionary of output messages keyed by model
        /// </summary>
        protected Dictionary<Model, DistributionType[]> Outputs = new Dictionary<Model, DistributionType[]>();

        /// <summary>
        /// Dictionary of variable copies indexed by model
        /// </summary>
        protected Dictionary<Model, Variable<DomainType>> variables = new Dictionary<Model, Variable<DomainType>>();

        /// <summary>
        /// Dictionary of priors indexed by model
        /// </summary>
        protected Dictionary<Model, Variable<DistributionType>> priors = new Dictionary<Model, Variable<DistributionType>>();

        /// <summary>
        /// Defining model - only one single-batch model can define a shared variable, and this is optional.
        /// </summary>
        protected Model DefiningModel = null;

        /// <summary>
        /// The algorithm
        /// </summary>
        protected IAlgorithm algorithm;

        /// <summary>
        /// Global counter used to generate variable names.
        /// </summary>
        private static readonly GlobalCounter globalCounter = new GlobalCounter();

        /// <summary>
        /// If true (the default), uses division to calculate the messages to batches.
        /// This is more efficient, but may introduce round-off errors.
        /// </summary>
        protected bool DivideMessages;

        internal SharedVariable(DistributionType prior, bool divideMessages = true)
        {
            this.Name = $"shared{StringUtil.TypeToString(typeof (DomainType))}{StringUtil.TypeToString(typeof (DistributionType))}{globalCounter.GetNext()}";
            this.Prior = prior;
            this.DivideMessages = divideMessages;
            if (divideMessages)
                this.CurrentMarginal = (DistributionType) this.Prior.Clone();
        }

        /// <summary>
        /// Constructs a new shared variable with a given domain type and distribution type
        /// </summary>
        /// <param name="name">Name of the shared variable</param>
        /// <returns></returns>
        public new SharedVariable<DomainType, DistributionType> Named(string name)
        {
            base.Named(name);
            return this;
        }

#if false
        SharedVariable<DomainType> SharedVariable<DomainType>.Named(string name)
        {
            return Named(name);
        }
#endif

        /// <summary>
        /// Gets a copy of this shared variable for the specified model
        /// </summary>
        /// <param name="model">The model identifier</param>
        /// <returns></returns>
        public override Variable<DomainType> GetCopyFor(Model model)
        {
            if (model == DefiningModel)
                throw new InferCompilerException("You cannot get a copy as the shared variable is defined by this model");

            Variable<DomainType> v;
            if (!variables.TryGetValue(model, out v))
            {
                Variable<DistributionType> vPrior = Variable.New<DistributionType>().Named(Name + "Prior");
                vPrior.ObservedValue = default(DistributionType);
                v = Variable<DomainType>.Random(vPrior).Named(Name).Attrib(QueryTypes.Marginal).Attrib(QueryTypes.MarginalDividedByPrior);
                variables[model] = v;
                model.SharedVariables.Add(this);
                priors[model] = vPrior;
                DistributionType[] messages = new DistributionType[model.BatchCount];
                for (int i = 0; i < messages.Length; i++)
                {
                    messages[i] = (DistributionType) Prior.Clone();
                    messages[i].SetToUniform();
                }
                if (DivideMessages)
                    CurrentMarginal = (DistributionType) Prior.Clone();
                Outputs[model] = messages;
            }
            return v;
        }

        /// <summary>
        /// Sets the definition of the shared variable
        /// </summary>
        /// <param name="model">Model id</param>
        /// <param name="definition">Defining variable</param>
        /// <returns></returns>
        /// <remarks>Use this method if the model is defining the shared variable rather than
        /// using one defined in this or another model.</remarks>
        public override void SetDefinitionTo(Model model, Variable<DomainType> definition)
        {
            if (DefiningModel != null)
                throw new InferCompilerException("You can only define a shared variable once");

            if (model.BatchCount != 1)
                throw new InferCompilerException("You can only define a shared variable from a model with a batch count of 1");

            if (!definition.IsBase)
                throw new InferCompilerException("You cannot set a shared variable to a derived variable");

            Variable<DomainType> v;
            if (!variables.TryGetValue(model, out v))
            {
                definition.AddAttribute(QueryTypes.Marginal);
                Variable<DistributionType> vPrior = Variable.New<DistributionType>().Named(Name + "Constraint");
                vPrior.ObservedValue = default(DistributionType);
                Variable.ConstrainEqualRandom<DomainType, DistributionType>(definition, vPrior);
                variables[model] = definition;
                model.SharedVariables.Add(this);
                priors[model] = vPrior;
                DistributionType[] messages = new DistributionType[model.BatchCount];
                for (int i = 0; i < messages.Length; i++)
                {
                    messages[i] = (DistributionType) Prior.Clone();
                    messages[i].SetToUniform();
                }
                if (DivideMessages)
                    CurrentMarginal = (DistributionType) Prior.Clone();
                // In this case, output refers to the forward message from the definition.
                // There is only one as we are requiring that batch count = 1.
                Outputs[model] = messages;

                // This is the defining model for the variable
                DefiningModel = model;
            }
        }

        /// <summary>
        /// Sets the shared variable's inbox given model and batch number
        /// </summary>
        /// <param name="model">Model id</param>
        /// <param name="batchNumber">Batch number</param>
        public override void SetInput(Model model, int batchNumber)
        {
            priors[model].ObservedValue = MessageToBatch(model, batchNumber);
            // this version mutates the ObservedValue in place.  unfortunately, if we do this the inference object will not detect that the value has changed.
            //priors[model].ObservedValue = MessageToBatch(model, batchNumber, priors[model].ObservedValue);
        }

        /// <summary>
        /// Returns the shared variable's inbox message given model and batch number
        /// </summary>
        /// <param name="model">Model id</param>
        /// <param name="batchNumber">Batch number</param>
        /// <returns>The inbox message</returns>
        public DistributionType MessageToBatch(Model model, int batchNumber)
        {
            return MessageToBatch(model, batchNumber, default(DistributionType));
        }

        /// <summary>
        /// Returns the shared variable's inbox message given model and batch number
        /// </summary>
        /// <param name="modelNumber">Model id</param>
        /// <param name="batchNumber">Batch number</param>
        /// <param name="result">Where to put the result</param>
        /// <returns>The inbox message</returns>
        public DistributionType MessageToBatch(Model modelNumber, int batchNumber, DistributionType result)
        {
            if (DivideMessages)
            {
                if (object.ReferenceEquals(result, default(DistributionType))) result = (DistributionType) CurrentMarginal.Clone();
                else result.SetTo(CurrentMarginal);
                if (algorithm == null) return result;

                foreach (KeyValuePair<Model, DistributionType[]> entry in Outputs)
                {
                    if (entry.Key == modelNumber)
                    {
                        // correct even for VMP
                        result.SetToRatio(result, entry.Value[batchNumber]);
                    }
                }
            }
            else
            {
                if (object.ReferenceEquals(result, default(DistributionType))) result = (DistributionType) Prior.Clone();
                else result.SetTo(Prior);
                if (algorithm == null) return result;
                foreach (KeyValuePair<Model, DistributionType[]> entry in Outputs)
                {
                    if (entry.Key == modelNumber)
                    {
                        // correct even for VMP
                        result = Distribution.SetToProductWithAllExcept(result, entry.Value, batchNumber);
                    }
                    else
                    {
                        result = Distribution.SetToProductWithAll(result, entry.Value);
                    }
                }
            }
            return result;
        }

        /// <summary>
        /// Get the marginal distribution, converted to type T
        /// </summary>
        /// <typeparam name="T">The desired type</typeparam>
        /// <returns></returns>
        public override T Marginal<T>()
        {
            return Distribution.ChangeType<T>(Marginal());
        }

        /// <summary>
        /// Returns the marginal distribution
        /// </summary>
        /// <returns></returns>
        public DistributionType Marginal()
        {
            return MessageToBatch(null, -1, default(DistributionType));
        }

        /// <summary>
        /// Gets the evidence correction for this shared variable
        /// </summary>
        /// <returns></returns>
        public override double GetEvidenceCorrection()
        {
            List<DistributionType> uses = new List<DistributionType>();
            foreach (DistributionType[] dists in Outputs.Values)
            {
                uses.AddRange(dists);
            }
            // this is correct for EP and VMP
            double result = UsesEqualDefOp.LogEvidenceRatio1(uses, Prior);
            if (DefiningModel != null)
            {
                if (!Prior.IsUniform())
                    throw new InferCompilerException("Shared variable has a non-uniform prior and a definition - try using a uniform prior instead");
                result -= Prior.GetLogAverageOf(Prior);
            }
            return result;
        }

        /// <summary>
        /// Infer the output message given a model id and a batch id
        /// </summary>
        /// <param name="engine">Inference engine</param>
        /// <param name="modelNumber">Model number</param>
        /// <param name="batchNumber">Batch number</param>
        public override void InferOutput(InferenceEngine engine, Model modelNumber, int batchNumber)
        {
            algorithm = engine.Algorithm;
            if (DivideMessages)
            {
                if (modelNumber != DefiningModel)
                {
                    Outputs[modelNumber][batchNumber] = (DistributionType) engine.GetOutputMessage<DistributionType>(GetCopyFor(modelNumber)).Clone();
                }
                else
                {
                    Outputs[modelNumber][batchNumber] = (DistributionType) engine.Infer<DistributionType>(variables[modelNumber]).Clone();
                    Outputs[modelNumber][batchNumber].SetToRatio(Outputs[modelNumber][batchNumber], priors[modelNumber].ObservedValue);
                }
                CurrentMarginal = engine.Infer<DistributionType>(variables[modelNumber]);
            }
            else
            {
                algorithm = engine.Algorithm;
                if (modelNumber != DefiningModel)
                {
                    Outputs[modelNumber][batchNumber] = engine.GetOutputMessage<DistributionType>(GetCopyFor(modelNumber));
                }
                else
                {
                    Outputs[modelNumber][batchNumber] = engine.Infer<DistributionType>(variables[modelNumber]);
                    Outputs[modelNumber][batchNumber].SetToRatio(Outputs[modelNumber][batchNumber], priors[modelNumber].ObservedValue);
                }
            }
        }

        /// <summary>
        /// Infer the output message given a model id and a batch id
        /// </summary>
        /// <param name="ca">Compiled algorithm</param>
        /// <param name="modelNumber">Model number</param>
        /// <param name="batchNumber">Batch number</param>
        public override void InferOutput(IGeneratedAlgorithm ca, Model modelNumber, int batchNumber)
        {
            throw new NotImplementedException();
            //if (algorithm is ExpectationPropagation && modelNumber != DefiningModel) {
            //    Outputs[modelNumber][batchNumber] = (DistributionType)ca.GetOutputMessage(Name);
            //} else {
            //    Outputs[modelNumber][batchNumber] = ca.Marginal<DistributionType>(variables[modelNumber].Name);
            //    // this can be avoided for VMP by labelling the variable as deterministic
            //    Outputs[modelNumber][batchNumber].SetToRatio(Outputs[modelNumber][batchNumber], priors[modelNumber].ObservedValue);
            //}
            //CurrentMarginal = ca.Marginal<DistributionType>(variables[modelNumber].Name);
        }
    }

    /// <summary>
    /// Interface for shared variables
    /// </summary>
    public interface ISharedVariable
    {
        /// <summary>
        /// Sets the shared variable's inbox for a given model and batch number
        /// </summary>
        /// <param name="modelNumber">Model id</param>
        /// <param name="batchNumber">Batch number</param>
        void SetInput(Model modelNumber, int batchNumber);

        /// <summary>
        /// Infers the shared variable's output message for a given model and batch number
        /// </summary>
        /// <param name="engine">Inference engine</param>
        /// <param name="modelNumber">Model id</param>
        /// <param name="batchNumber">Batch number</param>
        void InferOutput(InferenceEngine engine, Model modelNumber, int batchNumber);

        /// <summary>
        /// Infers the shared variable's output message for a given model and batch number
        /// </summary>
        /// <param name="ca">Compiled algorithm</param>
        /// <param name="modelNumber">Model id</param>
        /// <param name="batchNumber">Batch number</param>
        void InferOutput(IGeneratedAlgorithm ca, Model modelNumber, int batchNumber);

        /// <summary>
        /// Gets the evidence correction for this shared variable
        /// </summary>
        /// <returns></returns>
        double GetEvidenceCorrection();

        /// <summary>
        /// Whether this shared variable is an evidence variable
        /// </summary>
        bool IsEvidenceVariable { get; set; }
    }

    /// <summary>
    /// A Set of SharedVariables that allows SetInput/InferOutput to be called on all of them at once.
    /// </summary>
    public class SharedVariableSet : Set<ISharedVariable>, ISharedVariable
    {
        /// <summary>
        /// Constructs a set of shared variables
        /// </summary>
        public SharedVariableSet()
            : base()
        {
        }

#if false
        public SharedVariableSet(IEnumerable<ISharedVariable> variables) : base(variables)
        {
        }
#endif

        /// <summary>
        /// Set inboxes, for the given model and batch number, for all
        /// shared variables in this set
        /// </summary>
        /// <param name="modelNumber">Model id</param>
        /// <param name="batchNumber">Batch number</param>
        public void SetInput(Model modelNumber, int batchNumber)
        {
            foreach (ISharedVariable v in this)
            {
                v.SetInput(modelNumber, batchNumber);
            }
        }

        /// <summary>
        /// Infer the output messages, for the given model and batch number, for all
        /// shared variables in this set
        /// </summary>
        /// <param name="engine">Inference engine</param>
        /// <param name="modelNumber">Model id</param>
        /// <param name="batchNumber">Batch number</param>
        public void InferOutput(InferenceEngine engine, Model modelNumber, int batchNumber)
        {
            foreach (ISharedVariable v in this)
            {
                v.InferOutput(engine, modelNumber, batchNumber);
            }
        }

        /// <summary>
        /// Infer the output messages, for the given model and batch number, for all
        /// shared variables in this set
        /// </summary>
        /// <param name="ca">Compiled algorithm</param>
        /// <param name="modelNumber">Model id</param>
        /// <param name="batchNumber">Batch number</param>
        public void InferOutput(IGeneratedAlgorithm ca, Model modelNumber, int batchNumber)
        {
            foreach (ISharedVariable v in this)
            {
                v.InferOutput(ca, modelNumber, batchNumber);
            }
        }

        /// <summary>
        /// Gets the evidence for this set of shared variable
        /// </summary>
        /// <returns></returns>
        public double GetEvidence()
        {
            double sum = 0.0;
            foreach (ISharedVariable v in this)
            {
                if (!v.IsEvidenceVariable)
                    sum += v.GetEvidenceCorrection();
                else
                    sum += ((SharedVariable<bool, Bernoulli>) v).Marginal<Bernoulli>().LogOdds;
            }
            return sum;
        }

        /// <summary>
        /// Not supported for <see cref="SharedVariableSet"/>
        /// </summary>
        /// <returns></returns>
        public double GetEvidenceCorrection()
        {
            throw new NotSupportedException();
        }

        /// <summary>
        /// Not supported for <see cref="SharedVariableSet"/>
        /// </summary>
        public bool IsEvidenceVariable
        {
            get { return false; }
            set { throw new Exception("Cannot be an evidence variable"); }
        }
    }

    /// <summary>
    /// A model identifier used to manage SharedVariables.
    /// </summary>
    public class Model
    {
        /// <summary>
        /// The set of SharedVariables registered with this model.
        /// </summary>
        public SharedVariableSet SharedVariables = new SharedVariableSet();

        /// <summary>
        /// The number of data batches that will be processed with this model.
        /// </summary>
        public int BatchCount;

        /// <summary>
        /// Name of the model
        /// </summary>
        public string Name;

        private static readonly GlobalCounter globalCounter = new GlobalCounter();

        /// <summary>
        /// Create a new model identifier to which SharedVariables can be registered.
        /// </summary>
        /// <param name="batchCount">The number of data batches that will be processed with this model.</param>
        public Model(int batchCount)
        {
            Name = $"model{globalCounter.GetNext()}";
            BatchCount = batchCount;
        }

        /// <summary>
        /// Inline method for naming a shared variable model
        /// </summary>
        /// <param name="name"></param>
        /// <returns></returns>
        public Model Named(string name)
        {
            Name = name;
            return this;
        }

        /// <summary>
        /// Update all the SharedVariables registered with this model.
        /// </summary>
        /// <param name="engine"></param>
        /// <param name="batchNumber">A number from 0 to BatchCount-1</param>
        public void InferShared(InferenceEngine engine, int batchNumber)
        {
            SharedVariables.SetInput(this, batchNumber);
            SharedVariables.InferOutput(engine, this, batchNumber);
        }

        /// <summary>
        /// Update all the SharedVariables registered with this model.
        /// </summary>
        /// <param name="engine"></param>
        /// <param name="batchNumber">A number from 0 to BatchCount-1</param>
        public void InferShared(IGeneratedAlgorithm engine, int batchNumber)
        {
            SharedVariables.SetInput(this, batchNumber);
            SharedVariables.InferOutput(engine, this, batchNumber);
        }

        /// <summary>
        /// Gets evidence for all the specified models
        /// </summary>
        /// <param name="models">An array of models</param>
        /// <returns></returns>
        public static double GetEvidenceForAll(params Model[] models)
        {
            SharedVariableSet allVariables = new SharedVariableSet();
            foreach (Model model in models)
            {
                allVariables.AddRange(model.SharedVariables);
            }
            return allVariables.GetEvidence();
        }

        /// <summary>
        /// ToString override
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            if (BatchCount == 1) return Name;
            else return Name + "(" + BatchCount + ")";
        }
    }
}