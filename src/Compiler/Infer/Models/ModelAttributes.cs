// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Compiler;

namespace Microsoft.ML.Probabilistic.Models.Attributes
{
    /// <summary>
    /// Specifies the range of values taken by an integer variable, or the dimension of a Dirichlet variable.
    /// This attribute can be used to explicitly specify the value range for a variable
    /// in cases where it cannot be deduced by the model compiler.
    /// </summary>
    [AttributeUsage(AttributeTargets.All, AllowMultiple = false)]
    public class ValueRange : Attribute, ICompilerAttribute
    {
        /// <summary>
        /// The range indicating the values a variable can take or the dimension of the variable.
        /// </summary>
        public Range Range;

        /// <summary>
        /// Creates a ValueRange with the specified range.
        /// </summary>
        /// <param name="range"></param>
        public ValueRange(Range range)
        {
            this.Range = range;
        }

        /// <summary>
        /// Returns a string representation of the ValueRange.
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return "ValueRange(" + Range + ")";
        }
    }

    /// <summary>
    /// Specifies a prototype marginal distribution for a variable. This attribute
    /// can be used to explicitly specify the marginal distribution type for a variable
    /// in cases where it cannot be deduced by the model compiler.
    /// </summary>
    [AttributeUsage(AttributeTargets.All, AllowMultiple = false)]
    public class MarginalPrototype : Attribute, ICompilerAttribute
    {
        /// <summary>
        /// The prototype marginal distribution
        /// </summary>
        public object prototype;

        internal IExpression prototypeExpression;

        /// <summary>
        /// Creates a new marginal prototype attribute. This attribute
        /// targets variables.
        /// </summary>
        /// <param name="prototype">The marginal prototype</param>
        public MarginalPrototype(object prototype)
        {
            this.prototype = prototype;
        }

        /// <summary>
        /// ToString override
        /// </summary>
        /// <returns></returns>
        /// <exclude/>
        public override string ToString()
        {
            return "MarginalPrototype(" + ((prototypeExpression != null) ? prototypeExpression : prototype) + ")";
        }
    }

    ///// <summary>
    ///// Attribute which indicates a sparse marginal prototype
    ///// </summary>
    //public class Sparse : Attribute { }

    /// <summary>
    /// When attached to a Range, indicates that the elements of the range should be updated sequentially rather than in parallel.
    /// </summary>
    public class Sequential : ICompilerAttribute
    {
        /// <summary>
        /// If true, updates should be done in both directions of the loop
        /// </summary>
        public bool BackwardPass;
    }

    /// <summary>
    /// When attached to a Sequential Range, specifies which indices should be processed by each thread
    /// </summary>
    public class ParallelSchedule : ICompilerAttribute
    {
        internal IModelExpression scheduleExpression;

        /// <summary>
        /// Create a new ParallelSchedule attribute
        /// </summary>
        /// <param name="scheduleExpression">An observed variable of type int[][][], whose dimensions are [thread][block][item].  Each thread must have the same number of blocks, but blocks can be different sizes.  Must have at least one thread.</param>
        public ParallelSchedule(Variable<int[][][]> scheduleExpression)
        {
            this.scheduleExpression = scheduleExpression;
        }

        public override string ToString()
        {
            return "ParallelSchedule(" + scheduleExpression + ")";
        }
    }

    /// <summary>
    /// When attached to a Sequential Range, specifies which indices should be processed by each thread
    /// </summary>
    internal class ParallelScheduleExpression : ICompilerAttribute
    {
        internal IExpression scheduleExpression;

        public ParallelScheduleExpression(IExpression scheduleExpression)
        {
            this.scheduleExpression = scheduleExpression;
        }

        public override string ToString()
        {
            return "ParallelScheduleExpression(" + scheduleExpression + ")";
        }
    }

    /// <summary>
    /// When attached to a Sequential Range, specifies which indices should be processed by each thread
    /// </summary>
    public class DistributedSchedule : ICompilerAttribute
    {
        internal IModelExpression commExpression;
        internal IModelExpression scheduleExpression;
        internal IModelExpression schedulePerThreadExpression;

        /// <summary>
        /// Create a new DistributedSchedule attribute
        /// </summary>
        /// <param name="commExpression"></param>
        public DistributedSchedule(Variable<ICommunicator> commExpression)
        {
            this.commExpression = commExpression;
        }

        /// <summary>
        /// Create a new DistributedSchedule attribute
        /// </summary>
        /// <param name="commExpression"></param>
        /// <param name="scheduleExpression">An observed variable of type int[][], whose dimensions are [block][item].</param>
        public DistributedSchedule(Variable<ICommunicator> commExpression, Variable<int[][]> scheduleExpression)
        {
            this.commExpression = commExpression;
            this.scheduleExpression = scheduleExpression;
        }

        /// <summary>
        /// Create a new DistributedSchedule attribute
        /// </summary>
        /// <param name="commExpression"></param>
        /// <param name="schedulePerThreadExpression">An observed variable of type int[][][][], whose dimensions are [distributedStage][thread][block][item].  Each thread must have the same number of blocks, but blocks can be different sizes.  Must have at least one thread.</param>
        public DistributedSchedule(Variable<ICommunicator> commExpression, Variable<int[][][][]> schedulePerThreadExpression)
        {
            this.commExpression = commExpression;
            this.schedulePerThreadExpression = schedulePerThreadExpression;
        }

        public override string ToString()
        {
            return $"DistributedSchedule({scheduleExpression}, {schedulePerThreadExpression})";
        }
    }

    /// <summary>
    /// When attached to a Sequential Range, specifies which indices should be processed by each thread
    /// </summary>
    internal class DistributedScheduleExpression : ICompilerAttribute
    {
        internal IExpression commExpression;
        internal IExpression scheduleExpression;
        internal IExpression schedulePerThreadExpression;

        public DistributedScheduleExpression(IExpression commExpression, IExpression scheduleExpression, IExpression schedulePerThreadExpression)
        {
            this.commExpression = commExpression;
            this.scheduleExpression = scheduleExpression;
            this.schedulePerThreadExpression = schedulePerThreadExpression;
        }

        public override string ToString()
        {
            return $"DistributedScheduleExpression({scheduleExpression}, {schedulePerThreadExpression})";
        }
    }

    /// <summary>
    /// Attached to an index array.
    /// </summary>
    public class DistributedCommunication : ICompilerAttribute
    {
        internal IModelExpression arrayIndicesToSendExpression;
        internal IModelExpression arrayIndicesToReceiveExpression;

        /// <summary>
        /// Creates a new DistributedCommunication attribute
        /// </summary>
        /// <param name="arrayIndicesToSendExpression"></param>
        /// <param name="arrayIndicesToReceiveExpression"></param>
        public DistributedCommunication(IModelExpression arrayIndicesToSendExpression, IModelExpression arrayIndicesToReceiveExpression)
        {
            this.arrayIndicesToSendExpression = arrayIndicesToSendExpression;
            this.arrayIndicesToReceiveExpression = arrayIndicesToReceiveExpression;
        }

        public override string ToString()
        {
            return $"DistributedCommunication({arrayIndicesToSendExpression}, {arrayIndicesToReceiveExpression})";
        }
    }

    /// <summary>
    /// Attached to an index array.
    /// </summary>
    public class DistributedCommunicationExpression : ICompilerAttribute
    {
        internal IExpression arrayIndicesToSendExpression;
        internal IExpression arrayIndicesToReceiveExpression;

        /// <summary>
        /// Creates a new DistributedCommunication attribute
        /// </summary>
        /// <param name="arrayIndicesToSendExpression"></param>
        /// <param name="arrayIndicesToReceiveExpression"></param>
        public DistributedCommunicationExpression(IExpression arrayIndicesToSendExpression, IExpression arrayIndicesToReceiveExpression)
        {
            this.arrayIndicesToSendExpression = arrayIndicesToSendExpression;
            this.arrayIndicesToReceiveExpression = arrayIndicesToReceiveExpression;
        }

        public override string ToString()
        {
            return $"DistributedCommunicationExpression({arrayIndicesToSendExpression}, {arrayIndicesToReceiveExpression})";
        }
    }

    /// <summary>
    /// When attached to a Variable, specifies the initial forward messages to be used at the start of inference.
    /// </summary>
    internal class InitialiseTo : ICompilerAttribute
    {
        internal IExpression initialMessagesExpression;

        public InitialiseTo(IExpression initialMessagesExpression)
        {
            this.initialMessagesExpression = initialMessagesExpression;
        }

        public override string ToString()
        {
            return "InitialiseTo(" + initialMessagesExpression + ")";
        }
    }

    /// <summary>
    /// When attached to a Variable, specifies the initial backward messages to be used at the start of inference.
    /// </summary>
    internal class InitialiseBackwardTo : ICompilerAttribute
    {
        internal IExpression initialMessagesExpression;

        public InitialiseBackwardTo(IExpression initialMessagesExpression)
        {
            this.initialMessagesExpression = initialMessagesExpression;
        }

        public override string ToString()
        {
            return "InitialiseBackwardTo(" + initialMessagesExpression + ")";
        }
    }

    /// <summary>
    /// When attached to a Variable, indicates that the backward messages to factors with NoInit attributes should be treated as initialised by the scheduler, even though they will be initialised to uniform
    /// </summary>
    public class InitialiseBackward : ICompilerAttribute
    {
    }

    /// <summary>
    /// Attribute which associates a specified algorithm to a targetted variable or statement.
    /// This is used for hybrid inference where different algorithms are used for different parts
    /// of the model
    /// </summary>
    [AttributeUsage(AttributeTargets.All, AllowMultiple = false)]
    public class Algorithm : Attribute, ICompilerAttribute
    {
        /// <summary>
        /// The algorithm
        /// </summary>
        public IAlgorithm algorithm;

        /// <summary>
        /// Creates a new Algorithm attribute which assigns the given algorithm to the target
        /// </summary>
        /// <param name="algorithm"></param>
        public Algorithm(IAlgorithm algorithm)
        {
            this.algorithm = algorithm;
        }

        /// <summary>
        /// ToString override
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return string.Format("Algorithm({0})", algorithm);
        }
    }

    /// <summary>
    /// Attribute which associates a specified algorithm to all factors that define a variable.
    /// This is used for hybrid inference where different algorithms are used for different parts
    /// of the model
    /// </summary>
    public class FactorAlgorithm : ICompilerAttribute
    {
        /// <summary>
        /// The algorithm
        /// </summary>
        public IAlgorithm algorithm;

        /// <summary>
        /// Creates a new Algorithm attribute which assigns the given algorithm to the target's factor
        /// </summary>
        /// <param name="algorithm"></param>
        public FactorAlgorithm(IAlgorithm algorithm)
        {
            this.algorithm = algorithm;
        }

        /// <summary>
        /// ToString override
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return string.Format("FactorAlgorithm({0})", algorithm);
        }
    }

    /// <summary>
    /// Group member attribute - attached to MSL variables based on
    /// inference engine groups
    /// </summary>
    [AttributeUsage(AttributeTargets.All, AllowMultiple = true)]
    public class GroupMember : Attribute, ICompilerAttribute
    {
        /// <summary>
        /// The associated variable group
        /// </summary>
        public VariableGroup Group;

        /// <summary>
        /// This variable is a root in this group
        /// </summary>
        public bool IsRoot;

        /// <summary>
        /// Creates a group member attribute on a variable
        /// </summary>
        /// <param name="vg">The variable group</param>
        /// <param name="isRoot">Whether this variable is the root of the group</param>
        public GroupMember(VariableGroup vg, bool isRoot)
        {
            Group = vg;
            IsRoot = isRoot;
        }

        /// <summary>
        /// Returns a string representation of this group member attribute.
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            string strRoot = (IsRoot) ? " (root)" : "";
            return String.Format("GroupMember({0}{1})", Group, strRoot);
        }
    }

    /// <summary>
    /// When attached to a variable, indicates that the variable will not be inferred, producing more efficient generated code.
    /// </summary>
    public class DoNotInfer : Attribute, ICompilerAttribute
    {
    }

    /// <summary>
    /// For expert use only!  When sharing a variable between multiple models (e.g. using SharedVariable)
    /// you can add this attribute to have the variable be treated as a derived variable, even if it 
    /// is not derived in the submodel where it appears.
    /// </summary>
    public class DerivedVariable : Attribute, ICompilerAttribute
    {
    }

    /// <summary>
    /// Attribute to generate trace outputs for the messages associated with the target variable
    /// </summary>
    public class TraceMessages : Attribute, ICompilerAttribute
    {
        /// <summary>
        /// If non-null, only trace messages where the string representing the message expression
        /// contains this string.
        /// </summary>
        public string Containing { get; set; }
    }

    /// <summary>
    /// Attribute to cause message update events to be generated for the messages associated with the target variable
    /// </summary>
    public class ListenToMessages : Attribute, ICompilerAttribute
    {
        /// <summary>
        /// If non-null, only trace messages where the string representing the message expression
        /// contains this string.
        /// </summary>
        public string Containing { get; set; }
    }



    /// <summary>
    /// Attached to Variable or MethodInvoke to give priority in the operator search path
    /// </summary>
    public class GivePriorityTo : ICompilerAttribute
    {
        public object Container;

        public GivePriorityTo(object container)
        {
            this.Container = container;
        }

        public override string ToString()
        {
            return "GivePriorityTo(" + Container + ")";
        }
    }

    /// <summary>
    /// Attached to Variable objects to specify if outgoing messages should be computed by division
    /// </summary>
    public class DivideMessages : ICompilerAttribute
    {
        public bool useDivision;

        public DivideMessages(bool useDivision = true)
        {
            this.useDivision = useDivision;
        }

        public override string ToString()
        {
            return "DivideMessages(" + useDivision + ")";
        }
    }

    /// <summary>
    /// Attached to Ranges to specify that only one element should be in memory at a time (per thread)
    /// </summary>
    public class Partitioned : ICompilerAttribute
    {
    }

    /// <summary>
    /// Attached to Variable objects to indicate that their uncertainty should be ignored during inference.  
    /// The inferred marginal will always be a point mass.
    /// </summary>
    public class PointEstimate : ICompilerAttribute
    {
    }
}