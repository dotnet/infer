using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Probabilistic.Core.Maths
{
    struct TruncatedPowerSeries
    {
        public double[] Coefficients { get; }

        public TruncatedPowerSeries(double[] coefficients)
        {
            this.Coefficients = coefficients;
        }

        public static TruncatedPowerSeries Generate(int length, Func<int, double> coefGenerator)
        {
            double[] coefficients = new double[length];
            for (int i = 0; i < length; ++i)
                coefficients[i] = coefGenerator(i);

            return new TruncatedPowerSeries(coefficients);
        }

        public static TruncatedPowerSeries Generate(int length, Func<int, IList<double>, double> coefGenerator)
        {
            List<double> coefficients = new List<double>(length);
            for (int i = 0; i < length; ++i)
                coefficients.Add(coefGenerator(i, coefficients));

            return new TruncatedPowerSeries(coefficients.ToArray());
        }

        public double Evaluate(double x)
        {
            double sum = 0.0;
            for (int k = Coefficients.Length - 1; k > 0; k--)
            {
                sum = x * (Coefficients[k] + sum);
            }
            return sum + Coefficients[0];
        }
    }

    class PrecomputedSeriesCollection
    {
        public TruncatedPowerSeries GammaAt2 { get; }
        public TruncatedPowerSeries DigammaAt2 { get; }
        /// <summary>
        /// Coefficients of de Moivre's expansion for the digamma function.
        /// Each coefficient is B_{2j}/(2j) where B_{2j} are the Bernoulli numbers, starting from j=1
        /// </summary>
        public TruncatedPowerSeries CDigamma { get; }

        public PrecomputedSeriesCollection(uint precisionBits)
        {
            GammaAt2 = new TruncatedPowerSeries(new[]
            {
                0,
                0.32246703342411320303,
                -0.06735230105319810201,
                0.020580808427784546416,
                -0.0073855510286739856768,
                0.0028905103307415229257,
                -0.0011927539117032610189,
                0.00050966952474304234172,
                -0.00022315475845357938579,
                9.945751278180853098e-05,
                -4.4926236738133142046e-05,
                2.0507212775670691067e-05,
                -9.4394882752683967152e-06,
                4.3748667899074873274e-06,
                -2.0392157538013666132e-06,
                9.551412130407419353e-07,
                -4.4924691987645661855e-07,
                2.1207184805554664645e-07,
                -1.0043224823968100408e-07,
                4.7698101693639803983e-08,
                -2.2711094608943166813e-08,
                1.0838659214896952939e-08,
                -5.1834750419700474714e-09,
                2.4836745438024780616e-09,
                -1.1921401405860913615e-09,
                5.7313672416788612175e-10,
            });
            DigammaAt2 = TruncatedPowerSeries.Generate(GammaAt2.Coefficients.Length, i => GammaAt2.Coefficients[i] * (i + 1));
            CDigamma = new TruncatedPowerSeries(new[] {
                0, 1.0/12, -1.0/120, 1.0/252, -1.0/240, 1.0/132,
                -691.0/32760, 1.0/12, /* -3617.0/8160, 43867.0/14364, -174611.0/6600 */
            });
        }
    }
}
