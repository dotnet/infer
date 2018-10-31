// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Mappings
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Math;

    /// <summary>
    /// Represents the data splitting mapping that can be wrapped around any other recommender mapping to split the underlying data
    /// into the training and test parts.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
    /// <typeparam name="TInstance">The type of an instance.</typeparam>
    /// <typeparam name="TUser">The type of a user.</typeparam>
    /// <typeparam name="TItem">The type of an item.</typeparam>
    /// <typeparam name="TFeatureSource">The type of a feature source.</typeparam>
    /// <typeparam name="TFeatureValues">The type of the feature values.</typeparam>
    /// <remarks>
    /// <list type="number">
    /// <listheader>
    /// The data splitting algorithm works as follows:
    /// </listheader>
    /// <item>
    /// <description>
    /// The requested fraction of items is removed from the consideration (thus, ignored) 
    /// with all the associated observations.
    /// </description>
    /// </item>
    /// <item>
    /// <description>
    /// The requested fraction of items is put into the test set (so called cold items) 
    /// with all the associated observations.
    /// </description>
    /// </item>
    /// <item>
    /// <description>
    /// The requested fraction of the users remaining after (1-2) is removed from consideration 
    /// (thus, ignored) with all the associated observations.
    /// </description>
    /// </item>
    /// <item>
    /// <description>
    /// The requested fraction of the users remaining after (1-2) is put into the test set 
    /// (so called cold users) with all the associated observations.
    /// </description>
    /// </item>
    /// <item>
    /// <description>
    /// The requested fraction of the users remaining after (1-2) is put into 
    /// the training set with all the associated observations.
    /// </description>
    /// </item>
    /// <item>
    /// <description>
    /// For each user remaining after (1-5) the requested fraction of observations is put into the training set, while the 
    /// rest is put into the test set. At least one observation is always put into the training set for each user.
    /// </description>
    /// </item>
    /// <item>
    /// <description>
    /// If requested, instances associated with the occasionally produced cold items are removed from the test set.
    /// An item is said to be "occasionally cold" if it is covered only by the test set but wasn't explicitly chosen as cold.
    /// Such items can be produced by the steps (4) and (6). This option is supposed to be used when the requested number of 
    /// cold users and items is set to zero, in order to ensure that all the entities in the test set are covered by the training set.
    /// </description>
    /// </item>    
    /// </list>
    /// </remarks>
    public class TrainTestSplittingRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues> : 
        IRecommenderMapping<SplitInstanceSource<TInstanceSource>, TInstance, TUser, TItem, TFeatureSource, TFeatureValues>
    {
        /// <summary>
        /// The wrapped recommender mapping.
        /// </summary>
        private readonly IRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues> mapping;

        /// <summary>
        /// The fraction of users not included in either the training or the test set.
        /// </summary>
        private readonly double ignoredUserFraction;

        /// <summary>
        /// The fraction of items not included in either the training or the test set.
        /// </summary>
        private readonly double ignoredItemFraction;

        /// <summary>
        /// The fraction of users included in the training set only.
        /// </summary>
        private readonly double trainingOnlyUserFraction;

        /// <summary>
        /// The fraction of ratings in the training set for each user who is presented in both sets.
        /// </summary>
        private readonly double testUserRatingTrainingFraction;

        /// <summary>
        /// The fraction of users included in the test set only.
        /// </summary>
        private readonly double coldUserFraction;

        /// <summary>
        /// The fraction of items included in the test set only.
        /// </summary>
        private readonly double coldItemFraction;

        /// <summary>
        /// Whether the occasionally produced cold items should be removed from the test set.
        /// </summary>
        private readonly bool removeUndesiredColdItems;

        /// <summary>
        /// The last processed instance source.
        /// </summary>
        private TInstanceSource lastInstanceSource;

        /// <summary>
        /// Cached training set instances of the last processed instance source.
        /// </summary>
        private List<TInstance> cachedTrainingSetInstances;

        /// <summary>
        /// Cached test set instances of the last processed instance source.
        /// </summary>
        private List<TInstance> cachedTestSetInstances;

        /// <summary>
        /// Initializes a new instance of the
        /// <see cref="TrainTestSplittingRecommenderMapping{TInstanceSource,TInstance,TUser,TItem,TFeatureSource,TFeatureValues}"/> class.
        /// </summary>
        /// <param name="mapping">The wrapped recommender mapping.</param>
        /// <param name="trainingOnlyUserFraction">The fraction of users included in the training set only.</param>
        /// <param name="testUserRatingTrainingFraction">The fraction of ratings in the training set for each user who is presented in both sets.</param>
        /// <param name="coldUserFraction">The fraction of users included in the test set only.</param>
        /// <param name="coldItemFraction">The fraction of items included in the test set only.</param>
        /// <param name="ignoredUserFraction">The fraction of users not included in either the training or the test set.</param>
        /// <param name="ignoredItemFraction">The fraction of items not included in either the training or the test set.</param>
        /// <param name="removeOccasionalColdItems">Whether the occasionally produced cold items should be removed from the test set.</param>
        public TrainTestSplittingRecommenderMapping(
            IRecommenderMapping<TInstanceSource, TInstance, TUser, TItem, TFeatureSource, TFeatureValues> mapping,
            double trainingOnlyUserFraction,
            double testUserRatingTrainingFraction,
            double coldUserFraction = 0,
            double coldItemFraction = 0,
            double ignoredUserFraction = 0,
            double ignoredItemFraction = 0,
            bool removeOccasionalColdItems = false)
        {
            if (mapping == null)
            {
                throw new ArgumentNullException(nameof(mapping));
            }

            CheckFraction(trainingOnlyUserFraction, nameof(trainingOnlyUserFraction));
            CheckFraction(testUserRatingTrainingFraction, nameof(testUserRatingTrainingFraction));
            CheckFraction(coldUserFraction, nameof(coldUserFraction));
            CheckFraction(coldItemFraction, nameof(coldItemFraction));
            CheckFraction(ignoredUserFraction, nameof(ignoredUserFraction));
            CheckFraction(ignoredItemFraction, nameof(ignoredItemFraction));

            if (ignoredUserFraction + coldUserFraction + trainingOnlyUserFraction > 1)
            {
                throw new ArgumentException("The total fraction of cold users, training-only users and ignored users can not be greater than 1.");
            }

            if (ignoredItemFraction + coldItemFraction > 1)
            {
                throw new ArgumentException("The total fraction of cold items and ignored items can not be greater than 1.");
            }

            this.mapping = mapping;
            this.ignoredUserFraction = ignoredUserFraction;
            this.ignoredItemFraction = ignoredItemFraction;
            this.trainingOnlyUserFraction = trainingOnlyUserFraction;
            this.testUserRatingTrainingFraction = testUserRatingTrainingFraction;
            this.coldUserFraction = coldUserFraction;
            this.coldItemFraction = coldItemFraction;
            this.removeUndesiredColdItems = removeOccasionalColdItems;
        }

        /// <summary>
        /// Retrieves a list of instances from a given instance source.
        /// Dependent on <see cref="SplitInstanceSource{TInstanceSource}.IsTrainingSet"/>, the instances of 
        /// the training or test set are returned.
        /// </summary>
        /// <param name="instanceSource">The source to retrieve instances from.</param>
        /// <returns>The list of retrieved instances.</returns>
        public IEnumerable<TInstance> GetInstances(SplitInstanceSource<TInstanceSource> instanceSource)
        {
            if (instanceSource == null)
            {
                throw new ArgumentNullException(nameof(instanceSource));
            }
            
            this.UpdateCachedInstances(instanceSource.InstanceSource);
            return instanceSource.IsTrainingSet ? this.cachedTrainingSetInstances : this.cachedTestSetInstances;
        }

        /// <summary>
        /// Extracts a user from a given instance by delegating the call to the wrapped mapping.
        /// </summary>
        /// <param name="instanceSource">The source of instances providing the <paramref name="instance"/>.</param>
        /// <param name="instance">The instance to extract the user from.</param>
        /// <returns>The extracted user.</returns>
        public TUser GetUser(SplitInstanceSource<TInstanceSource> instanceSource, TInstance instance)
        {
            if (instanceSource == null)
            {
                throw new ArgumentNullException(nameof(instanceSource));
            }
            
            return this.mapping.GetUser(instanceSource.InstanceSource, instance);
        }

        /// <summary>
        /// Extracts an item from a given instance by delegating the call to the wrapped mapping.
        /// </summary>
        /// <param name="instanceSource">The source of instances providing the <paramref name="instance"/>.</param>
        /// <param name="instance">The instance to extract the item from.</param>
        /// <returns>The extracted item.</returns>
        public TItem GetItem(SplitInstanceSource<TInstanceSource> instanceSource, TInstance instance)
        {
            if (instanceSource == null)
            {
                throw new ArgumentNullException(nameof(instanceSource));
            }
            
            return this.mapping.GetItem(instanceSource.InstanceSource, instance);
        }

        /// <summary>
        /// Provides the features for a given user by delegating the call to the wrapped mapping.
        /// </summary>
        /// <param name="featureSource">The source of the features.</param>
        /// <param name="user">The user to provide the features for.</param>
        /// <returns>The features for <paramref name="user"/>.</returns>
        public TFeatureValues GetUserFeatures(TFeatureSource featureSource, TUser user)
        {
            return this.mapping.GetUserFeatures(featureSource, user);
        }

        /// <summary>
        /// Provides the features for a given item by delegating the call to the wrapped mapping.
        /// </summary>
        /// <param name="featureSource">The source of the features.</param>
        /// <param name="item">The item to provide the features for.</param>
        /// <returns>The features for <paramref name="item"/>.</returns>
        public TFeatureValues GetItemFeatures(TFeatureSource featureSource, TItem item)
        {
            return this.mapping.GetItemFeatures(featureSource, item);
        }

        /// <summary>
        /// Checks if the specified value is between zero and one.
        /// </summary>
        /// <param name="fraction">The value.</param>
        /// <param name="parameterName">The name of the value to use in the exception message.</param>
        /// <exception cref="ArgumentOutOfRangeException">Thrown if <paramref name="fraction"/> is not between zero and one.</exception>
        private static void CheckFraction(double fraction, string parameterName)
        {
            if (fraction < 0 || fraction > 1)
            {
                throw new ArgumentOutOfRangeException(parameterName, "The value of this parameter should be between zero and one.");
            }
        }

        /// <summary>
        /// Adds the given element to a list indexed by a given key. If key is not presented in the dictionary,
        /// a new list is created and added to it.
        /// </summary>
        /// <typeparam name="TKey">The type of the dictionary key.</typeparam>
        /// <typeparam name="TListElement">The type of the list element.</typeparam>
        /// <param name="dictionary">The dictionary.</param>
        /// <param name="key">The key.</param>
        /// <param name="listElement">The element to add.</param>
        private static void AddToDictionaryOfLists<TKey, TListElement>(
            Dictionary<TKey, List<TListElement>> dictionary, TKey key, TListElement listElement)
        {
            List<TListElement> list;
            if (!dictionary.TryGetValue(key, out list))
            {
                list = new List<TListElement>();
                dictionary.Add(key, list);
            }

            list.Add(listElement);
        }

        /// <summary>
        /// Splits the given instance source into the training and test set and updates the cached lists of instances.
        /// </summary>
        /// <param name="instanceSource">The instance source to split.</param>
        private void UpdateCachedInstances(TInstanceSource instanceSource)
        {
            if (object.Equals(instanceSource, this.lastInstanceSource))
            {
                return; // No need to rebuild
            }
            
            this.lastInstanceSource = instanceSource;
            this.cachedTrainingSetInstances = new List<TInstance>();
            this.cachedTestSetInstances = new List<TInstance>();

            IEnumerable<TInstance> instances = this.mapping.GetInstances(instanceSource);

            // Build mapping from items to lists of corresponding instances
            var itemToInstances = new Dictionary<TItem, List<TInstance>>();
            foreach (TInstance instance in instances)
            {
                TItem item = this.mapping.GetItem(instanceSource, instance);
                AddToDictionaryOfLists(itemToInstances, item, instance);
            }

            // Randomly permute the set of items
            TItem[] items = itemToInstances.Keys.ToArray();
            Rand.Shuffle(items);

            // Split instances on per-item basis
            int coldItemCount = Convert.ToInt32(items.Length * this.coldItemFraction);
            int ignoredItemCount = Convert.ToInt32(items.Length * this.ignoredItemFraction);
            Debug.Assert(coldItemCount + ignoredItemCount <= items.Length, "Inconsistent counts of ignored and cold items.");
            var remainingInstances = new List<TInstance>();
            for (int i = 0; i < items.Length; ++i)
            {
                IEnumerable<TInstance> itemInstances = itemToInstances[items[i]];
                if (i < coldItemCount)
                {
                    this.cachedTestSetInstances.AddRange(itemInstances); // Instances associated with cold items go to test set
                }
                else if (i >= coldItemCount + ignoredItemCount)
                {
                    remainingInstances.AddRange(itemInstances); // Instances associated with items which are not cold or ignored remain
                }

                // All the other items are ignored
            }

            // Build mapping from users to lists of corresponding instances using remaining instances only
            var userToInstances = new Dictionary<TUser, List<TInstance>>();
            Rand.Shuffle(remainingInstances);
            foreach (TInstance instance in remainingInstances)
            {
                TUser user = this.mapping.GetUser(instanceSource, instance);
                AddToDictionaryOfLists(userToInstances, user, instance);
            }

            // Randomly permute the set of users
            TUser[] users = userToInstances.Keys.ToArray();
            Rand.Shuffle(users);

            List<TInstance> testInstances = this.cachedTestSetInstances;
            HashSet<TItem> itemsInTrainingSet = null;
            if (this.removeUndesiredColdItems)
            {
                // If the occasional cold item filtering is enabled,
                // we will create the list of test instances in two steps, filtering out any observations with undesired cold items
                testInstances = new List<TInstance>();
                itemsInTrainingSet = new HashSet<TItem>();
            }

            // Split instances on per-user basis
            int coldUserCount = Convert.ToInt32(users.Length * this.coldUserFraction);
            int ignoredUserCount = Convert.ToInt32(users.Length * this.ignoredUserFraction);
            int trainingOnlyUserCount = Convert.ToInt32(users.Length * this.trainingOnlyUserFraction);
            Debug.Assert(
                coldUserCount + ignoredUserCount + trainingOnlyUserCount <= users.Length,
                "Inconsistent counts of ignored, cold and training-only users.");
            for (int i = 0; i < users.Length; ++i)
            {
                List<TInstance> userInstances = userToInstances[users[i]];
                if (i < coldUserCount)
                {
                    testInstances.AddRange(userInstances); // Instances associated with cold users go to test set
                }
                else if (i < coldUserCount + trainingOnlyUserCount)
                {
                    this.cachedTrainingSetInstances.AddRange(userInstances); // Instances associated with training-only users go to training set
                    if (this.removeUndesiredColdItems)
                    {
                        itemsInTrainingSet.UnionWith(userInstances.Select(instance => this.mapping.GetItem(instanceSource, instance)));
                    }
                }
                else if (i >= coldUserCount + trainingOnlyUserCount + ignoredUserCount)
                {
                    // Instances of the remaining users are split between training and test sets
                    int trainingInstanceCount = Convert.ToInt32(userInstances.Count * this.testUserRatingTrainingFraction);
                    trainingInstanceCount = Math.Max(trainingInstanceCount, 1); // Always take at least one instance for each user
                    this.cachedTrainingSetInstances.AddRange(userInstances.Take(trainingInstanceCount));
                    testInstances.AddRange(userInstances.Skip(trainingInstanceCount));
                    if (this.removeUndesiredColdItems)
                    {
                        itemsInTrainingSet.UnionWith(
                            userInstances.Take(trainingInstanceCount).Select(instance => this.mapping.GetItem(instanceSource, instance)));
                    }
                }

                // All the other users are ignored
            }

            // If the occasional cold item filtering is enabled, we should add to test set only instances with items covered in the training set
            if (this.removeUndesiredColdItems)
            {
                this.cachedTestSetInstances.AddRange(
                    testInstances.Where(instance => itemsInTrainingSet.Contains(this.mapping.GetItem(instanceSource, instance))));
            }
        }
    }
}
