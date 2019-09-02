using Microsoft.ML.Probabilistic.Math;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RobustGaussianProcess
{
    class Utilities
    {
        // Train Gaussian Process on the small 'Auto Insurance in Sweden' dataset
        // Add the insurance.csv file to a folder named Data
        private const string AisCsvPath = @"..\..\..\Data\insurance.csv";

        // Path for the results plot
        private const string OutputPlotPath = @"..\..\..\Data\GPRegressionPredictions.png";

        /// <summary>
        /// Generates a 1D vector with length len having a min and max; data points are randomly distributed and ordered if specified
        /// </summary>
        public static Vector[] VectorRange(double min, double max, int len, bool random)
        {
            Vector[] inputs = new Vector[len];
            Random rng = new Random();

            for (int i = 0; i < len; i++)
            {
                double num = new double();

                if (random)
                {
                    num = rng.NextDouble();
                }
                else
                {
                    num = i / (double)(len - 1);
                }
                num = num * (max - min);
                num += min;
                inputs[i] = Vector.FromArray(new double[1] { num });
            }

            return inputs;
        }

        public static IEnumerable<(double x, double y)> LoadAISDataset()
        {
            var data = new List<(double x, double y)>();

            // Read CSV file
            using (var reader = new StreamReader(AisCsvPath))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line.Split(',');
                    if (values.Length == 2)
                    {
                        data.Add((double.Parse(values[0]), double.Parse(values[1])));
                    }
                }
            }

            return PreprocessData(data).ToList();
        }

        public static IEnumerable<(double x, double y)> PreprocessData(
            IEnumerable<(double x, double y)> data)
        {
            var x = data.Select(tup => tup.x);
            var y = data.Select(tup => tup.y);

            // Shift targets so mean is 0
            var meanY = y.Sum() / y.Count();
            y = y.Select(val => val - meanY);

            // Scale data to lie between 1 and -1
            var absoluteMaxY = y.Select(val => System.Math.Abs(val)).Max();
            y = y.Select(val => val / absoluteMaxY);
            var maxX = x.Max();
            x = x.Select(val => val / maxX);
            var dataset = x.Zip(y, (a, b) => (a, b));

            // Order data by input value
            return dataset.OrderBy(tup => tup.Item1);
        }
    }
}
