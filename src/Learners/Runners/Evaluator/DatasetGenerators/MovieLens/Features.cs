// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners.MovieLens
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Text;

    /// <summary>
    /// Provides methods for computing feature vectors.
    /// </summary>
    public static class Features
    {
        /// <summary>
        /// The number of buckets in the user gender feature.
        /// </summary>
        public static readonly int UserGenderBucketCount = 2;

        /// <summary>
        /// The number of buckets in the user age feature.
        /// </summary>
        public static readonly int UserAgeBucketCount = 7;

        /// <summary>
        /// The number of buckets in the user occupation feature.
        /// </summary>
        public static readonly int UserOccupationBucketCount = 21;

        /// <summary>
        /// The number of buckets in the movie year feature.
        /// </summary>
        public static readonly int MovieYearBucketCount = 24;

        /// <summary>
        /// The number of buckets in the movie genre feature.
        /// </summary>
        public static readonly int MovieGenreBucketCount = 18;

        /// <summary>
        /// The user ages delimiting age intervals.
        /// </summary>
        private static readonly int[] UserAges = { 1, 18, 25, 35, 45, 50, 56 };

        /// <summary>
        /// All possible movie genres.
        /// </summary>
        private static readonly string[] MovieGenreNames =
        {
            "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
            "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
            "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
        };

        /// <summary>
        /// The movie years delimiting year intervals.
        /// </summary>
        private static readonly int[] MovieYears;

        /// <summary>
        /// The mapping between movie genre name and feature bucket.
        /// </summary>
        private static readonly Dictionary<string, int> MovieGenreBuckets;

        /// <summary>
        /// Initializes static members of the <see cref="Features"/> class.
        /// </summary>
        static Features()
        {
            // Movie years
            MovieYears = new int[MovieYearBucketCount];
            int bucket = 0;

            for (int i = 1930; i < 1960; i += 10)
            {
                MovieYears[bucket++] = i;
            }

            for (int i = 1960; i < 1985; i += 5)
            {
                MovieYears[bucket++] = i;
            }

            for (int i = 1985; i <= 2000; ++i)
            {
                MovieYears[bucket++] = i;
            }

            Debug.Assert(bucket == MovieYearBucketCount, "The number of assigned buckets should be equal to the overall number of buckets.");

            // Movie genre buckets
            MovieGenreBuckets = new Dictionary<string, int>();
            for (int i = 0; i < MovieGenreNames.Length; ++i)
            {
                MovieGenreBuckets[MovieGenreNames[i]] = i;
            }
        }

        /// <summary>
        /// Computes the bucket values of the user gender feature.
        /// </summary>
        /// <param name="offset">The current number of buckets in the global feature vector.</param>
        /// <param name="feature">The feature to process.</param>
        /// <returns>The number of feature buckets and the computed feature vector.</returns>
        public static Tuple<int, string> ComputeUserGender(int offset, string feature)
        {
            int bucket;

            if (string.Compare(feature, "M", StringComparison.OrdinalIgnoreCase) == 0)
            {
                bucket = 0;
            }
            else if (string.Compare(feature, "F", StringComparison.OrdinalIgnoreCase) == 0)
            {
                bucket = 1;
            }
            else
            {
                throw new ArgumentException(string.Format("User gender should be either 'M' or 'F'; given {0}", feature));
            }

            return new Tuple<int, string>(UserGenderBucketCount, string.Format("{0}:1", offset + bucket));
        }

        /// <summary>
        /// Computes the bucket values of the user age feature.
        /// </summary>
        /// <param name="offset">The current number of buckets in the global feature vector.</param>
        /// <param name="feature">The feature to process.</param>
        /// <returns>The number of feature buckets and the computed feature vector.</returns>
        public static Tuple<int, string> ComputeUserAge(int offset, string feature)
        {
            int bucket = -1;
            int age;

            if (!int.TryParse(feature, out age))
            {
                throw new ArgumentException(string.Format("Invalid age: {0}.", feature));
            }

            int i;
            for (i = 0; i < UserAges.Length; ++i)
            {
                if (age == UserAges[i])
                {
                    bucket = i;
                    break;
                }
            }

            if (i == UserAges.Length)
            {
                throw new ArgumentException(string.Format("Invalid age: {0}.", feature));
            }

            return new Tuple<int, string>(UserAgeBucketCount, string.Format("{0}:1", offset + bucket));
        }

        /// <summary>
        /// Computes the bucket values of the user occupation feature.
        /// </summary>
        /// <param name="offset">The current number of buckets in the global feature vector.</param>
        /// <param name="feature">The feature to process.</param>
        /// <returns>The number of feature buckets and the computed feature vector.</returns>
        public static Tuple<int, string> ComputeUserOccupation(int offset, string feature)
        {
            int occupation;

            if (!int.TryParse(feature, out occupation))
            {
                throw new ArgumentException(string.Format("Invalid occupation: {0}.", feature));
            }

            if (occupation < 0 || occupation > 20)
            {
                throw new ArgumentException(string.Format("Occupation should be between 0 and 20; given {0}.", occupation));
            }

            return new Tuple<int, string>(UserOccupationBucketCount, string.Format("{0}:1", offset + occupation));
        }

        /// <summary>
        /// Computes the bucket values of the movie year feature.
        /// </summary>
        /// <param name="offset">The current number of buckets in the global feature vector.</param>
        /// <param name="feature">The feature to process.</param>
        /// <returns>The number of feature buckets and the computed feature vector.</returns>
        public static Tuple<int, string> ComputeMovieYear(int offset, string feature)
        {
            // Extract the year from the movie title
            feature = feature.Substring(feature.Length - 5, 4);

            int year;

            if (!int.TryParse(feature, out year))
            {
                throw new ArgumentException(string.Format("Invalid movie release year: {0}.", feature));
            }

            if (year < 1919 || year > 2000)
            {
                throw new ArgumentException(string.Format("The movie release year should be between 1919 and 2000; given {0}", feature));
            }

            int bucket = 0;
            while (year > MovieYears[bucket])
            {
                ++bucket;
            }

            string result; // interpolate
            if (bucket > 0 && MovieYears[bucket] != year)
            {
                // set two values
                double years = MovieYears[bucket] - MovieYears[bucket - 1];

                double rvalue = (year - MovieYears[bucket - 1]) / years;
                double lvalue = 1 - rvalue;

                result = string.Format("{0}:{1}|{2}:{3}", offset + bucket - 1, lvalue, offset + bucket, rvalue);
            }
            else
            {
                // set only one value
                result = string.Format("{0}:1", offset + bucket);
            }

            return new Tuple<int, string>(MovieYearBucketCount, result);
        }

        /// <summary>
        /// Computes the bucket values of the movie year genre.
        /// </summary>
        /// <param name="offset">The current number of buckets in the global feature vector.</param>
        /// <param name="feature">The feature to process.</param>
        /// <returns>The number of feature buckets and the computed feature vector.</returns>
        public static Tuple<int, string> ComputeMovieGenre(int offset, string feature)
        {
            string[] genres = feature.Split('|');
            // genres.Length will always be at least 1
            double value = 1.0 / genres.Length;

            var result = new StringBuilder(string.Format("{0}:{1}", offset + MovieGenreBuckets[genres[0]], value));
            for (int i = 1; i < genres.Length; ++i)
            {
                result.Append(string.Format("|{0}:{1}", offset + MovieGenreBuckets[genres[i].Trim()], value));
            }

            return new Tuple<int, string>(MovieGenreBucketCount, result.ToString());
        }
    }
}
