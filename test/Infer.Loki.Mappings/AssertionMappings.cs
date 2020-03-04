using Loki.Mapping;
using Loki.Shared;
using System;
using System.Numerics.MPFR;

namespace Infer.Loki.Mappings
{
    public static class AssertionMappings
    {
        public static readonly BigFloat TOLERANCE = BigFloatFactory.Create(2e-30);

        public static bool IsErrorSignificantPreciser(BigFloat assertTolerance, BigFloat error)
        {
            var smallerTolerance = BigFloatFactory.Empty();
            using (var coef = BigFloatFactory.Create(2e-15))
                BigFloat.Mul(smallerTolerance, assertTolerance, coef);
            return BigFloat.Greater(error, smallerTolerance);
        }
    }
}
