// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;

    /// <summary>
    /// Utility functions to serialize and deserialize recommender predictions of various types to text files.
    /// </summary>
    public static class RecommenderPersistenceUtils
    {
        /// <summary>
        /// Saves rating predictions to a CSV file.
        /// </summary>
        /// <param name="fileName">The name of the file to save rating predictions to.</param>
        /// <param name="predictedRatings">The rating predictions to save.</param>
        public static void SavePredictedRatings(string fileName, IDictionary<User, IDictionary<Item, int>> predictedRatings)
        {
            if (predictedRatings == null)
            {
                throw new ArgumentNullException(nameof(predictedRatings));
            }

            if (string.IsNullOrEmpty(fileName))
            {
                throw new ArgumentException("A valid file name should be provided.", nameof(fileName));
            }
            
            using (var writer = new StreamWriter(fileName))
            {
                foreach (KeyValuePair<User, IDictionary<Item, int>> userWithPredictedRatings in predictedRatings)
                {
                    foreach (KeyValuePair<Item, int> itemWithRating in userWithPredictedRatings.Value)
                    {
                        writer.WriteLine("{0},{1},{2}", userWithPredictedRatings.Key.Id, itemWithRating.Key.Id, itemWithRating.Value);
                    }
                }
            }
        }

        /// <summary>
        /// Loads rating predictions from a CSV file.
        /// </summary>
        /// <param name="fileName">The name of the file to load rating predictions from.</param>
        /// <returns>The loaded rating predictions.</returns>
        /// <exception cref="InvalidFileFormatException">Thrown if the file has invalid format.</exception>
        public static IDictionary<User, IDictionary<Item, int>> LoadPredictedRatings(string fileName)
        {
            if (string.IsNullOrEmpty(fileName))
            {
                throw new ArgumentException("A valid file name should be provided.", nameof(fileName));
            }
            
            using (var reader = new StreamReader(fileName))
            {
                var result = new Dictionary<User, IDictionary<Item, int>>();
                
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    string[] parts = line.Split(',');

                    int rating;
                    string userId, itemId;
                    if (parts.Length != 3 ||
                        (userId = parts[0].Trim()).Length == 0 ||
                        (itemId = parts[1].Trim()).Length == 0 ||
                        !int.TryParse(parts[2], out rating))
                    {
                        throw new InvalidFileFormatException(string.Format("Rating prediction file '{0}' has invalid format.", fileName));
                    }

                    var user = new User(userId, null);
                    var item = new Item(itemId, null);

                    IDictionary<Item, int> itemToRating;
                    if (!result.TryGetValue(user, out itemToRating))
                    {
                        itemToRating = new Dictionary<Item, int>();
                        result.Add(user, itemToRating);
                    }

                    if (itemToRating.ContainsKey(item))
                    {
                        throw new InvalidFileFormatException(
                            string.Format("More than one rating for user {0} and item {1} in rating prediction file '{2}'.", userId, itemId, fileName));
                    }

                    itemToRating.Add(item, rating);
                }

                return result;
            }
        }

        /// <summary>
        /// Saves item recommendations to a text file.
        /// </summary>
        /// <param name="fileName">The name of the file to save item recommendations to.</param>
        /// <param name="recommendations">The item recommendations.</param>
        public static void SaveRecommendedItems(string fileName, IDictionary<User, IEnumerable<Item>> recommendations)
        {
            if (recommendations == null)
            {
                throw new ArgumentNullException(nameof(recommendations));
            }

            if (string.IsNullOrEmpty(fileName))
            {
                throw new ArgumentException("A valid file name should be provided.", nameof(fileName));
            }
            
            SaveEntityLists(fileName, recommendations, u => u.Id, i => i.Id);
        }

        /// <summary>
        /// Loads item recommendations from a text file.
        /// </summary>
        /// <param name="fileName">The name of the file to load the item recommendations from.</param>
        /// <returns>The loaded item recommendations.</returns>
        /// <exception cref="InvalidFileFormatException">Thrown if the file has invalid format.</exception>
        public static IDictionary<User, IEnumerable<Item>> LoadRecommendedItems(string fileName)
        {
            if (string.IsNullOrEmpty(fileName))
            {
                throw new ArgumentException("A valid file name should be provided.", nameof(fileName));
            }
            
            return LoadEntityLists(id => new User(id, null), id => new Item(id, null), fileName);
        }

        /// <summary>
        /// Saves lists of related users to a text file.
        /// </summary>
        /// <param name="fileName">The name of the file to save related users to.</param>
        /// <param name="relatedUsers">The dictionary which maps a user to a list of related users.</param>
        public static void SaveRelatedUsers(string fileName, IDictionary<User, IEnumerable<User>> relatedUsers)
        {
            if (relatedUsers == null)
            {
                throw new ArgumentNullException(nameof(relatedUsers));
            }

            if (string.IsNullOrEmpty(fileName))
            {
                throw new ArgumentException("A valid file name should be provided.", nameof(fileName));
            }
            
            SaveEntityLists(fileName, relatedUsers, u => u.Id, u => u.Id);
        }

        /// <summary>
        /// Loads lists of related users from a text file.
        /// </summary>
        /// <param name="fileName">The name of the file to load the lists of related users from.</param>
        /// <returns>The lists of related users.</returns>
        /// <exception cref="InvalidFileFormatException">Thrown if the file has invalid format.</exception>
        public static IDictionary<User, IEnumerable<User>> LoadRelatedUsers(string fileName)
        {
            if (string.IsNullOrEmpty(fileName))
            {
                throw new ArgumentException("A valid file name should be provided.", nameof(fileName));
            }
            
            Func<string, User> userFactory = id => new User(id, null);
            return LoadEntityLists(userFactory, userFactory, fileName);
        }

        /// <summary>
        /// Saves lists of related items to a text file.
        /// </summary>
        /// <param name="fileName">The name of the file to save related items to.</param>
        /// <param name="relatedItems">The dictionary which maps an item to a list of related items.</param>
        public static void SaveRelatedItems(string fileName, IDictionary<Item, IEnumerable<Item>> relatedItems)
        {
            if (relatedItems == null)
            {
                throw new ArgumentNullException(nameof(relatedItems));
            }

            if (string.IsNullOrEmpty(fileName))
            {
                throw new ArgumentException("A valid file name should be provided.", nameof(fileName));
            }
            
            SaveEntityLists(fileName, relatedItems, i => i.Id, i => i.Id);
        }

        /// <summary>
        /// Loads lists of related items from a text file.
        /// </summary>
        /// <param name="fileName">The name of the file to load the lists of related users from.</param>
        /// <returns>The lists of related users.</returns>
        /// <exception cref="InvalidFileFormatException">Thrown if the file has invalid format.</exception>
        public static IDictionary<Item, IEnumerable<Item>> LoadRelatedItems(string fileName)
        {
            if (string.IsNullOrEmpty(fileName))
            {
                throw new ArgumentException("A valid file name should be provided.", nameof(fileName));
            }
            
            Func<string, Item> itemFactory = id => new Item(id, null);
            return LoadEntityLists(itemFactory, itemFactory, fileName);
        }

        /// <summary>
        /// Saves a collection of entity lists to a text file.
        /// </summary>
        /// <typeparam name="TEntity">The type of an entity that acts as a header of a list.</typeparam>
        /// <typeparam name="TEntityInList">The type of entities in list.</typeparam>
        /// <param name="fileName">The name of the file to save the lists to.</param>
        /// <param name="entityLists">The mapping from list head to list contents.</param>
        /// <param name="entityIdSelector">The function to get the identifier of a list head.</param>
        /// <param name="entityInListIdSelector">The function to get the identifier of a list member.</param>
        private static void SaveEntityLists<TEntity, TEntityInList>(
            string fileName,
            IDictionary<TEntity, IEnumerable<TEntityInList>> entityLists,
            Func<TEntity, string> entityIdSelector,
            Func<TEntityInList, string> entityInListIdSelector)
        {
            Debug.Assert(entityLists != null, "A valid entity list dictionary should be provided.");
            Debug.Assert(entityIdSelector != null && entityInListIdSelector != null, "Valid entity id selectors should be provided.");
            Debug.Assert(!string.IsNullOrEmpty(fileName), "A valid file name should be provided.");
            
            using (var writer = new StreamWriter(fileName))
            {
                foreach (var entityWithList in entityLists)
                {
                    TEntity entity = entityWithList.Key;
                    IEnumerable<TEntityInList> entityList = entityWithList.Value;

                    writer.Write(entityIdSelector(entity));
                    bool first = true;
                    foreach (TEntityInList entityInList in entityList)
                    {
                        writer.Write(first ? ':' : ',');
                        writer.Write(entityInListIdSelector(entityInList));
                        first = false;
                    }

                    writer.WriteLine();
                }
            }
        }

        /// <summary>
        /// Loads a collection of entity lists from a text file.
        /// </summary>
        /// <typeparam name="TEntity">The type of an entity that acts as a header of a list.</typeparam>
        /// <typeparam name="TEntityInList">The type of entities in list.</typeparam>
        /// <param name="entityFactory">The factory to produce list heads from identifiers.</param>
        /// <param name="entityInListFactory">The factory to produce list members from identifiers.</param>
        /// <param name="fileName">The name of the file to load the entity lists from.</param>
        /// <returns>The loaded entity lists represented as a mapping from list head to list contents.</returns>
        /// <exception cref="InvalidFileFormatException">Thrown if the file has invalid format.</exception>
        private static IDictionary<TEntity, IEnumerable<TEntityInList>> LoadEntityLists<TEntity, TEntityInList>(
            Func<string, TEntity> entityFactory,
            Func<string, TEntityInList> entityInListFactory,
            string fileName)
        {
            Debug.Assert(entityFactory != null && entityInListFactory != null, "Valid entity factories should be provided.");
            Debug.Assert(!string.IsNullOrEmpty(fileName), "A valid file name should be provided.");
            
            using (var reader = new StreamReader(fileName))
            {
                var result = new Dictionary<TEntity, IEnumerable<TEntityInList>>();

                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    string[] lineParts = line.Split(':');
                    string entityId, listString;
                    if (lineParts.Length != 2 ||
                        (entityId = lineParts[0].Trim()).Length == 0 ||
                        (listString = lineParts[1].Trim()).Length == 0)
                    {
                        throw new InvalidFileFormatException(
                            string.Format("Each line in the file '{0}' should contain a non-empty identifier and a list of identifiers separated by colon.", fileName));
                    }

                    var entity = entityFactory(entityId);
                    if (result.ContainsKey(entity))
                    {
                        throw new InvalidFileFormatException(
                            string.Format("More than one set of list for {0} in the file '{1}'", entity, fileName));
                    }

                    var entityList = new List<TEntityInList>();
                    var entityInListSet = new HashSet<TEntityInList>();
                    string[] listStringParts = listString.Split(',');
                    for (int i = 0; i < listStringParts.Length; ++i)
                    {
                        string entityInListId = listStringParts[i].Trim();
                        if (entityInListId.Length == 0)
                        {
                            throw new InvalidFileFormatException(
                                string.Format("Empty identifier found in one of the lists in the file '{0}'", fileName));
                        }
                        
                        var entityInList = entityInListFactory(entityInListId);
                        if (entityInListSet.Contains(entityInList))
                        {
                            throw new InvalidFileFormatException(
                                string.Format("{0} occured more than once in the list for {1} in the file '{2}'", entityInListId, entityId, fileName));
                        }

                        entityList.Add(entityInList);
                        entityInListSet.Add(entityInList);
                    }

                    result.Add(entity, entityList);
                }

                return result;
            }
        }
    }
}
