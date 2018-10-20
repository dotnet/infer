using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Probabilistic.Learners.Mappings;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Learners.Tests
{
    using Microsoft.ML.Probabilistic.Learners.Runners;
    using Xunit;
    using Assert = AssertHelper;

    [Serializable]
    public class CsvMapping : IStarRatingRecommenderMapping<string, Tuple<string, string, int>, string, string, int, NoFeatureSource, Vector>
    {
        public IEnumerable<Tuple<string, string, int>> GetInstances(string instanceSource)
        {
            foreach (string line in File.ReadLines(instanceSource))
            {
                string[] split = line.Split(new[] { ',' });
                yield return Tuple.Create(split[0], split[1], Convert.ToInt32(split[2]));
            }
        }

        public string GetUser(string instanceSource, Tuple<string, string, int> instance)
        { return instance.Item1; }

        public string GetItem(string instanceSource, Tuple<string, string, int> instance)
        { return instance.Item2; }

        public int GetRating(string instanceSource, Tuple<string, string, int> instance)
        { return instance.Item3; }

        public IStarRatingInfo<int> GetRatingInfo(string instanceSource)
        { return new StarRatingInfo(0, 5); }

        public Vector GetUserFeatures(NoFeatureSource featureSource, string user)
        { throw new NotImplementedException(); }

        public Vector GetItemFeatures(NoFeatureSource featureSource, string item)
        { throw new NotImplementedException(); }
    }

    public class CsvMappingTests
    {
        /// <summary>
        /// Tests the sample code at https://dotnet.github.io/infer/userguide/Learners/Matchbox%20recommender/Learner%20API.html
        /// </summary>
        [Fact]
        public void CsvMappingExample()
        {
            GenerateCsvFile("Ratings.csv");
            var dataMapping = new CsvMapping();
            var recommender = MatchboxRecommender.Create(dataMapping);
            recommender.Settings.Training.TraitCount = 5;
            recommender.Settings.Training.IterationCount = 20;
            recommender.Train("Ratings.csv");
            var recommendations = recommender.Recommend("Person 1", 10);
            CheckRecommendations(recommendations);
        }

        [Fact]
        public void RecommenderDatasetExample()
        {
            GenerateRecommenderDatasetFile("RatingsDataset.csv");
            RecommenderDataset trainingDataset = RecommenderDataset.Load("RatingsDataset.csv");
            var recommender = MatchboxRecommender.Create(Mappings.StarRatingRecommender);
            recommender.Settings.Training.TraitCount = 5;
            recommender.Settings.Training.IterationCount = 20;
            recommender.Train(trainingDataset);
            var recommendations = recommender.Recommend(new User("Person 1", Vector.FromArray(2.3)), 10);
            CheckRecommendations(recommendations.Select(item => item.Id));
        }

        private void GenerateCsvFile(string filename)
        {
            using (var writer = new StreamWriter(filename))
            {
                GenerateData(writer);
            }
        }

        private void GenerateRecommenderDatasetFile(string filename)
        {
            using (var writer = new StreamWriter(filename))
            {
                writer.WriteLine("R,1,5");
                GenerateData(writer);
            }
        }

        private void GenerateData(StreamWriter writer)
        {
            // each line is:
            // Person Name, Movie Name, Rating 1-5
            for (int person = 0; person < 20; person++)
            {
                for (int observation = 0; observation < 5; observation++)
                {
                    int movie = person + observation;
                    int personType = person % 2;
                    int movieType = movie % 2;
                    int rating = (personType == movieType) ? 5 : 1;
                    writer.WriteLine($"Person {person},Movie {movie},{rating}");
                }
            }
        }

        private void CheckRecommendations(IEnumerable<string> recommendations)
        {
            foreach (var movie in recommendations)
            {
                // Person 1 should like odd-numbered movies
                int movieIndex = int.Parse(movie.Split(' ')[1]);
                Assert.True(movieIndex % 2 == 1);
            }
        }
    }
}
