// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Mappings
{
    using System.Collections.Generic;

    /// <summary>
    /// A mapping used by the implementations of
    /// <see cref="IMatchboxRecommender{TInstanceSource, TUser, TItem, TRatingDistribution, TFeatureSource}"/>
    /// interface to convert user data to an appropriate format.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
    /// <typeparam name="TFeatureSource">The type of a source of features.</typeparam>
    /// <remarks>
    /// Feature-related methods in this interface are only called when user of item features are being used. 
    /// During training, features for all entities are required, and hence the methods called are the GetAll*Feature* ones.
    /// During prediction, features are requested on per-entity basis, and hence the GetSingle*Feature* methods are called.
    /// Note that entity features are expected not to change between training and prediction, and therefore only features
    /// of cold users or items will be requested during prediction.
    /// </remarks>
    public interface IMatchboxRecommenderMapping<in TInstanceSource, in TFeatureSource>
    {
        /// <summary>
        /// Gets the list of user identifiers from a given instance source.
        /// </summary>
        /// <param name="instanceSource">The source of instances to get the user identifiers from.</param>
        /// <param name="batchNumber">The number of the current batch (used only if the data is divided into batches).</param>
        /// <returns>The list of user identifiers.</returns>
        /// <remarks>
        /// Note that this method can be called concurrently from multiple threads, 
        /// so its implementation must be thread-safe.
        /// </remarks>
        IReadOnlyList<int> GetUserIds(TInstanceSource instanceSource, int batchNumber = 0);

        /// <summary>
        /// Gets the list of item identifiers from a given instance source.
        /// </summary>
        /// <param name="instanceSource">The source of instances to get the item identifiers from.</param>
        /// <param name="batchNumber">The number of the current batch (used only if the data is divided into batches).</param>
        /// <returns>The list of item identifiers.</returns>
        /// <remarks>
        /// Note that this method can be called concurrently from multiple threads, 
        /// so its implementation must be thread-safe.
        /// </remarks>
        IReadOnlyList<int> GetItemIds(TInstanceSource instanceSource, int batchNumber = 0);

        /// <summary>
        /// Gets the list of ratings from a given instance source.
        /// </summary>
        /// <param name="instanceSource">The source of instances to get the ratings from.</param>
        /// <param name="batchNumber">The number of the current batch (used only if the data is divided into batches).</param>
        /// <returns>The list of ratings</returns>
        /// <remarks>
        /// Note that this method can be called concurrently from multiple threads, 
        /// so its implementation must be thread-safe.
        /// </remarks>
        IReadOnlyList<int> GetRatings(TInstanceSource instanceSource, int batchNumber = 0);

        /// <summary>
        /// Gets the number of users from a given instance source.
        /// </summary>
        /// <param name="instanceSource">The source of instances to get number of users from.</param>
        /// <returns>The number of users.</returns>
        int GetUserCount(TInstanceSource instanceSource);

        /// <summary>
        /// Gets the number of items from a given instance source.
        /// </summary>
        /// <param name="instanceSource">The source of instances to get number of items from.</param>
        /// <returns>The number of items.</returns>
        int GetItemCount(TInstanceSource instanceSource);

        /// <summary>
        /// Gets the number of star ratings.
        /// This is equal to one plus the difference between the maximum and the minimum rating.
        /// </summary>
        /// <param name="instanceSource">The source of instances to get number of items from.</param>
        /// <returns>The number of ratings.</returns>
        int GetRatingCount(TInstanceSource instanceSource);

        /// <summary>
        /// Gets the number of user features.
        /// </summary>
        /// <param name="featureSource">The source to obtain features from.</param>
        /// <returns>The number of user features.</returns>
        int GetUserFeatureCount(TFeatureSource featureSource);

        /// <summary>
        /// Gets the number of item features.
        /// </summary>
        /// <param name="featureSource">The source to obtain features from.</param>
        /// <returns>The number of item features.</returns>
        int GetItemFeatureCount(TFeatureSource featureSource);

        /// <summary>
        /// Gets non-zero feature values for all users present in a given feature source.
        /// </summary>
        /// <param name="featureSource">The source to obtain features from.</param>
        /// <returns>An array of non-zero user feature arrays where the outer array is indexed by user id.</returns>
        /// <remarks>This function will be called during training if the user feature support is enabled.</remarks>
        IReadOnlyList<IReadOnlyList<double>> GetAllUserNonZeroFeatureValues(TFeatureSource featureSource);

        /// <summary>
        /// Gets non-zero feature values for all items present in a given feature source.
        /// </summary>
        /// <param name="featureSource">The source to obtain features from.</param>
        /// <returns>An array of non-zero item feature arrays where the outer array is indexed by item id</returns>
        /// <remarks>This function will be called during training if the item feature support is enabled.</remarks>
        IReadOnlyList<IReadOnlyList<double>> GetAllItemNonZeroFeatureValues(TFeatureSource featureSource);

        /// <summary>
        /// Gets non-zero feature indices for all users present in a given feature source.
        /// </summary>
        /// <param name="featureSource">The source to obtain feature indices from.</param>
        /// <returns>An array of non-zero user feature index arrays where the outer array is indexed by user id.</returns>
        /// <remarks>This function will be called during training if the user feature support is enabled.</remarks>
        IReadOnlyList<IReadOnlyList<int>> GetAllUserNonZeroFeatureIndices(TFeatureSource featureSource);

        /// <summary>
        /// Gets non-zero feature indices for all items present in a given feature source.
        /// </summary>
        /// <param name="featureSource">The source to obtain feature indices from.</param>
        /// <returns>An array of non-zero item feature index arrays where the outer array is indexed by item id</returns>
        /// <remarks>This function will be called during training if the item feature support is enabled.</remarks>
        IReadOnlyList<IReadOnlyList<int>> GetAllItemNonZeroFeatureIndices(TFeatureSource featureSource);

        /// <summary>
        /// Gets non-zero feature values for a given user.
        /// </summary>
        /// <param name="featureSource">The source to obtain features from.</param>
        /// <param name="userId">The user identifier.</param>
        /// <returns>Non-zero feature values for the user.</returns>
        /// <remarks>This function will be called during prediction for cold users if the user feature support is enabled.</remarks>
        IReadOnlyList<double> GetSingleUserNonZeroFeatureValues(TFeatureSource featureSource, int userId);

        /// <summary>
        /// Gets non-zero feature values for a given item.
        /// </summary>
        /// <param name="featureSource">The source to obtain features from.</param>
        /// <param name="itemId">The item identifier.</param>
        /// <returns>Non-zero feature values for the item.</returns>
        /// <remarks>This function will be called during prediction for cold items if the item feature support is enabled.</remarks>
        IReadOnlyList<double> GetSingleItemNonZeroFeatureValues(TFeatureSource featureSource, int itemId);

        /// <summary>
        /// Gets non-zero feature indices for a given user.
        /// </summary>
        /// <param name="featureSource">The source to obtain feature indices from.</param>
        /// <param name="userId">The user identifier.</param>
        /// <returns>Non-zero feature indices for the user.</returns>
        /// <remarks>This function will be called during prediction for cold users if the user feature support is enabled.</remarks>
        IReadOnlyList<int> GetSingleUserNonZeroFeatureIndices(TFeatureSource featureSource, int userId);

        /// <summary>
        /// Gets non-zero feature indices for a given item.
        /// </summary>
        /// <param name="featureSource">The source to obtain feature indices from.</param>
        /// <param name="itemId">The item identifier.</param>
        /// <returns>Non-zero feature values for the item.</returns>
        /// <remarks>This function will be called during prediction for cold items if the item feature support is enabled.</remarks>
        IReadOnlyList<int> GetSingleItemNonZeroFeatureIndices(TFeatureSource featureSource, int itemId);
    }
}