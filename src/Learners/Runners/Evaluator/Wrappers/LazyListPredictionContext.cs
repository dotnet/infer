// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System.Collections;
    using System.Collections.Generic;
    using System.Diagnostics;

    /// <summary>
    /// Stores a bunch of list prediction queries (such as item recommendation/related items/related users queries),
    /// which are executed when the results of some query are needed for the first time.
    /// Can be useful to batch multiple queries together and execute them in a single call to the recommender.
    /// </summary>
    /// <typeparam name="TQueryEntity">The type of an entity acting as a query.</typeparam>
    /// <typeparam name="TResultEntity">The type of a query result entity.</typeparam>
    internal abstract class LazyListPredictionContext<TQueryEntity, TResultEntity>
    {
        /// <summary>
        /// The set of query entities for the queries added to this context so far.
        /// </summary>
        private readonly HashSet<TQueryEntity> queryEntitiesInContext = new HashSet<TQueryEntity>();
            
        /// <summary>
        /// The computed predictions.
        /// </summary>
        private IDictionary<TQueryEntity, IEnumerable<TResultEntity>> predictions;

        /// <summary>
        /// Initializes a new instance of the <see cref="LazyListPredictionContext{TQueryEntity, TResultEntity}"/> class.
        /// </summary>
        /// <param name="maxPredictionListSize">The maximum size of the prediction list.</param>
        protected LazyListPredictionContext(int maxPredictionListSize)
        {
            this.MaxPredictionListSize = maxPredictionListSize;
        }

        /// <summary>
        /// Gets the maximum size of the prediction list.
        /// </summary>
        public int MaxPredictionListSize { get; private set; }

        /// <summary>
        /// Checks if a new context must be created for a given entity and a maximum prediction list size.
        /// New context may be needed if the current one has been evaluated already,
        /// some query for a given entity is already presented in the current context,
        /// or maximum prediction list size is different from the one associated with the context.
        /// </summary>
        /// <param name="queryEntity">The entity to make predictions for.</param>
        /// <param name="maxPredictionListSize">The maximum size of the prediction list.</param>
        /// <returns>True if a new context should be created, false otherwise.</returns>
        public bool IsNewContextNeeded(TQueryEntity queryEntity, int maxPredictionListSize)
        {
            return this.predictions != null || this.queryEntitiesInContext.Contains(queryEntity) || this.MaxPredictionListSize != maxPredictionListSize;
        }
        
        /// <summary>
        /// Creates a prediction list wrapper for a given query entity. Enumeration of the wrapper will result in the evaluation of the context.
        /// </summary>
        /// <param name="queryEntity">The entity to make predictions for.</param>
        /// <param name="possibleResultEntities">The set of entities that can possibly be presented in the prediction.</param>
        /// <returns>The created prediction list wrapper.</returns>
        public IEnumerable<TResultEntity> CreateLazyPredictionResults(TQueryEntity queryEntity, IEnumerable<TResultEntity> possibleResultEntities)
        {
            Debug.Assert(this.predictions == null, "New queries can't be added into the evaluated context.");
            Debug.Assert(!this.queryEntitiesInContext.Contains(queryEntity), "Entity can have only one query associated with it in every context.");

            this.queryEntitiesInContext.Add(queryEntity);
            this.AppendQuery(queryEntity, possibleResultEntities);
            return new LazyPredictionsCollection(this, queryEntity);
        }

        /// <summary>
        /// Overridden in the derived classes to append a given query to the context.
        /// </summary>
        /// <param name="queryEntity">The entity to make predictions for.</param>
        /// <param name="possibleResultEntities">The set of entities that can possibly be presented in the prediction.</param>
        protected abstract void AppendQuery(TQueryEntity queryEntity, IEnumerable<TResultEntity> possibleResultEntities);

        /// <summary>
        /// Overridden in the derived classes to compute the predictions for all the queries stored in the context.
        /// </summary>
        /// <returns>The list of predictions for every query entity stored in the context.</returns>
        protected abstract IDictionary<TQueryEntity, IEnumerable<TResultEntity>> EvaluateContext();
        
        /// <summary>
        /// Gets the prediction list for a given entity. If the context has not been evaluated yet, it will be.
        /// </summary>
        /// <param name="queryEntity">The entity to make predictions for.</param>
        /// <returns>The list of predictions for the query entity.</returns>
        private IEnumerable<TResultEntity> GetPredictionsForEntity(TQueryEntity queryEntity)
        {
            Debug.Assert(this.queryEntitiesInContext.Contains(queryEntity), "A query for the requested entity should have been added to the context before.");

            if (this.predictions == null)
            {
                this.predictions = this.EvaluateContext();
            }

            return this.predictions[queryEntity];
        }

        /// <summary>
        /// Represents results of a list prediction query computed on demand.
        /// </summary>
        private class LazyPredictionsCollection : IEnumerable<TResultEntity>
        {
            /// <summary>
            /// The context associated with the corresponding query.
            /// </summary>
            private readonly LazyListPredictionContext<TQueryEntity, TResultEntity> context;

            /// <summary>
            /// The query entity.
            /// </summary>
            private readonly TQueryEntity queryEntity;

            /// <summary>
            /// Initializes a new instance of the <see cref="LazyPredictionsCollection"/> class.
            /// </summary>
            /// <param name="context">The context associated with the corresponding query..</param>
            /// <param name="queryEntity">The query entity.</param>
            public LazyPredictionsCollection(LazyListPredictionContext<TQueryEntity, TResultEntity> context, TQueryEntity queryEntity)
            {
                this.context = context;
                this.queryEntity = queryEntity;
            }

            /// <summary>
            /// Returns an enumerator that iterates through the collection. Triggers the evaluation of the associated context.
            /// </summary>
            /// <returns>An enumerator.</returns>
            public IEnumerator<TResultEntity> GetEnumerator()
            {
                IEnumerable<TResultEntity> recommendations = this.context.GetPredictionsForEntity(this.queryEntity);
                return recommendations.GetEnumerator();
            }

            /// <summary>
            /// Returns an enumerator that iterates through the collection. Triggers the evaluation of the associated context.
            /// </summary>
            /// <returns>An enumerator.</returns>
            IEnumerator IEnumerable.GetEnumerator()
            {
                return this.GetEnumerator();
            }
        }
    }
}
