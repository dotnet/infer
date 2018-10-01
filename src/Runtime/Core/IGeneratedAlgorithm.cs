// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Probabilistic
{
    /// <summary>
    /// Interface for running a generated inference algorithm
    /// </summary>
    public interface IGeneratedAlgorithm
    {
        /// <summary>
        /// The number of iterations done from the initial state or since the last change of observed values
        /// </summary>
        /// <remarks>
        /// Changing an observed value will reset this to 0.
        /// </remarks>
        int NumberOfIterationsDone { get; }

        /// <summary>
        /// Execute the inference algorithm for the specified number of iterations, starting from the initial state
        /// </summary>
        /// <param name="numberOfIterations">The number of iterations to perform from the initial state</param>
        /// <remarks>
        /// Sets <c>NumberOfIterationsDone = <paramref name="numberOfIterations"/></c>.
        /// This method is equivalent to calling <c>Reset()</c> followed by <c>Update(numberOfIterations)</c>.
        /// </remarks>
        void Execute(int numberOfIterations);

        /// <summary>
        /// Perform additional iterations of the inference algorithm
        /// </summary>
        /// <param name="additionalIterations">The number of additional iterations to perform</param>
        /// <remarks>
        /// If no observed values have changed, this method increments <c>NumberOfIterationsDone</c> by <paramref name="additionalIterations"/>,
        /// and is equivalent to calling <c>Execute(NumberOfIterationsDone + additionalIterations)</c>.
        /// If some observed values have changed, this method sets <c>NumberOfIterationsDone = <paramref name="additionalIterations"/></c>,
        /// and is not equivalent to calling <c>Execute</c>, because it will start from the existing message state rather than the initial state.
        /// </remarks>
        void Update(int additionalIterations);

        /// <summary>
        /// Reset all messages to their initial values.  Sets NumberOfIterationsDone to 0
        /// </summary>
        /// <remarks>
        /// This method is equivalent to calling <c>Execute(0)</c>
        /// </remarks>
        void Reset();

        /// <summary>
        /// Get the marginal distribution (computed up to this point) of a variable
        /// </summary>
        /// <param name="variableName">Name of the variable in the generated code</param>
        /// <returns>The marginal distribution computed up to this point</returns>
        /// <remarks>
        /// Execute, Update, or Reset must be called first to set the value of the marginal.
        /// </remarks>
        object Marginal(string variableName);

        /// <summary>
        /// Get the marginal distribution (computed up to this point) of a variable, converted to type T
        /// </summary>
        /// <typeparam name="T">The distribution type.</typeparam>
        /// <param name="variableName">Name of the variable in the generated code</param>
        /// <returns>The marginal distribution computed up to this point</returns>
        /// <remarks>
        /// Execute, Update, or Reset must be called first to set the value of the marginal.
        /// The conversion operation may be costly, even if the marginal already has type T.
        /// For maximum efficiency, use the non-generic Marginal method when conversion is not needed.
        /// </remarks>
        T Marginal<T>(string variableName);

        /// <summary>
        /// Get the query-specific marginal distribution of a variable.
        /// For example, GibbsSampling answers "Marginal", "Samples", and "Conditionals" queries
        /// </summary>
        /// <param name="variableName">Name of the variable in the generated code</param>
        /// <param name="query">Query string</param>
        /// <returns>The query-specific marginal distribution computed up to this point</returns>
        /// <remarks>
        /// Execute, Update, or Reset must be called first to set the value of the marginal.
        /// </remarks>
        object Marginal(string variableName, string query);

        /// <summary>
        /// Get the query-specific marginal distribution of a variable, converted to type T
        /// </summary>
        /// <typeparam name="T">Type to cast to</typeparam>
        /// <param name="variableName">Name of the variable in the generated code</param>
        /// <param name="query">Query</param>
        /// <returns>The query-specific marginal distribution computed up to this point</returns>
        /// <remarks>
        /// Execute, Update, or Reset must be called first to set the value of the marginal.
        /// The conversion operation may be costly, even if the marginal already has type T.
        /// For maximum efficiency, use the non-generic Marginal method when conversion is not needed.
        /// </remarks>
        T Marginal<T>(string variableName, string query);

        /// <summary>
        /// Gets an observed value
        /// </summary>
        /// <param name="variableName">Name of the variable in the generated code</param>
        /// <returns>The observed value</returns>
        object GetObservedValue(string variableName);

        /// <summary>
        /// Sets an observed value
        /// </summary>
        /// <param name="variableName">Name of the variable in the generated code</param>
        /// <param name="value">The observed value</param>
        void SetObservedValue(string variableName, object value);

        /// <summary>
        /// Fired when the progress of inference changes, typically at the
        /// end of one iteration of the inference algorithm
        /// </summary>
        event EventHandler<ProgressChangedEventArgs> ProgressChanged;
    }
}