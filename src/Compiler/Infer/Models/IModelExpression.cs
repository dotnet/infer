// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Models
{
    /// <summary>
    /// Interface to a modelling expression, such as a constant, variable or parameter.
    /// </summary>
    public interface IModelExpression
    {
        /// <summary>
        /// Get the code model expression
        /// </summary>
        /// <returns></returns>
        IExpression GetExpression();

        /// <summary>
        /// Expression name
        /// </summary>
        string Name { get; }
    }

    /// <summary>
    /// Generic inferface to a modelling expression of type T.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    public interface IModelExpression<T> : IModelExpression
    {
    }

    /// <summary>
    /// A marker interface for variables.
    /// </summary>
    public interface IVariable : IModelExpression, HasObservedValue
    {
    }

    /// <summary>
    /// Interface for getting list of containers
    /// </summary>
    public interface CanGetContainers
    {
        /// <summary>
        /// Get list of containers for a variable
        /// </summary>
        /// <typeparam name="T">Type of variable</typeparam>
        /// <returns></returns>
        List<T> GetContainers<T>();
    }

    /// <summary>
    /// Interface for a variable to have an observed value
    /// </summary>
    public interface HasObservedValue
    {
        /// <summary>
        /// Returns true if the variable is observed.
        /// </summary>
        bool IsObserved { get; }

        /// <summary>
        /// Observed value property
        /// </summary>
        object ObservedValue { get; set; }
    }
}