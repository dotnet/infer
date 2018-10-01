// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;

    /// <summary>
    /// Represents a matrix of values for predicted ratings versus true ratings.
    /// Usages include a confusion matrix and a loss matrix.
    /// </summary>
    public class RatingMatrix
    {
        /// <summary>
        /// The values of each (predicted rating, true rating) pair.
        /// </summary>
        private readonly double[,] data;

        /// <summary>
        /// Initializes a new instance of the <see cref="RatingMatrix"/> class.
        /// </summary>
        /// <param name="minRating">The minimum rating.</param>
        /// <param name="maxRating">The maximum rating.</param>
        public RatingMatrix(int minRating, int maxRating)
        {
            int size = maxRating - minRating + 1;
            this.data = new double[size, size];
            this.MinRating = minRating;
            this.MaxRating = maxRating;
        }

        /// <summary>
        /// Gets the minimum rating.
        /// </summary>
        public int MinRating { get; private set; }

        /// <summary>
        /// Gets the maximum rating.
        /// </summary>
        public int MaxRating { get; private set; }

        /// <summary>
        /// Gets or sets the value of a specified rating pair.
        /// </summary>
        /// <param name="predictedRating">The predicted rating value.</param>
        /// <param name="trueRating">The true rating value.</param>
        /// <returns>The value at the specified pair of indices.</returns>
        /// <remarks>The specified rating indices should be in the interval [min rating; max rating].</remarks>
        public double this[int predictedRating, int trueRating]
        {
            get
            {
                return this.data[predictedRating - this.MinRating, trueRating - this.MinRating];
            }

            set
            {
                this.data[predictedRating - this.MinRating, trueRating - this.MinRating] = value;
            }
        }

        /// <summary>
        /// Computes the component-wise products of two rating matrices.
        /// </summary>
        /// <param name="lhs">The left-hand side matrix.</param>
        /// <param name="rhs">The right-hand side matrix.</param>
        /// <returns>The sum of the products of the corresponding elements of the two matrices.</returns>
        public static double ComponentwiseProduct(RatingMatrix lhs, RatingMatrix rhs)
        {
            if (lhs.MinRating != rhs.MinRating || lhs.MaxRating != rhs.MaxRating)
            {
                throw new ArgumentException("The given matrices are incompatible!");
            }

            double result = 0;
            for (int i = lhs.MinRating; i <= lhs.MaxRating; ++i)
            {
                for (int j = lhs.MinRating; j <= lhs.MaxRating; ++j)
                {
                    result += lhs[i, j] * rhs[i, j];
                }
            }

            return result;
        }

        /// <summary>
        /// Generates an absolute error loss matrix. That is, the value of each element 
        /// is equal to the absolute difference of the indices of the element.
        /// </summary>
        /// <param name="minRating">The minimum rating.</param>
        /// <param name="maxRating">The maximum rating.</param>
        /// <returns>The generated absolute error loss matrix.</returns>
        public static RatingMatrix AbsoluteErrorLossMatrix(int minRating, int maxRating)
        {
            return LossMatrix(minRating, maxRating, (i, j) => Math.Abs(i - j));
        }

        /// <summary>
        /// Generates a squared error loss matrix. That is, the value of each element 
        /// is equal to the squared difference of the indices of the element.
        /// </summary>
        /// <param name="minRating">The minimum rating.</param>
        /// <param name="maxRating">The maximum rating.</param>
        /// <returns>The generated squared error loss matrix.</returns>
        public static RatingMatrix SquaredErrorLossMatrix(int minRating, int maxRating)
        {
            return LossMatrix(minRating, maxRating, (i, j) => (i - j) * (i - j));
        }

        /// <summary>
        /// Generates a zero-one error loss matrix. That is, the value of each element 
        /// is equal to zero if its indices are equal, or one otherwise.
        /// </summary>
        /// <param name="minRating">The minimum rating.</param>
        /// <param name="maxRating">The maximum rating.</param>
        /// <returns>The generated squared error loss matrix.</returns>
        public static RatingMatrix ZeroOneErrorLossMatrix(int minRating, int maxRating)
        {
            return LossMatrix(minRating, maxRating, (i, j) => i == j ? 0.0 : 1.0);
        }

        /// <summary>
        /// Generates a loss matrix from a given loss function.
        /// </summary>
        /// <param name="minRating">The minimum rating.</param>
        /// <param name="maxRating">The maximum rating.</param>
        /// <param name="lossFunc">The loss function, which operates on index pairs.</param>
        /// <returns>The generated loss matrix.</returns>
        private static RatingMatrix LossMatrix(int minRating, int maxRating, Func<int, int, double> lossFunc)
        {
            var result = new RatingMatrix(minRating, maxRating);

            for (int i = 1; i <= maxRating; ++i)
            {
                for (int j = 1; j <= maxRating; ++j)
                {
                    result[i, j] = lossFunc(i, j);
                }
            }

            return result;
        }
    }
}