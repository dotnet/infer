using Loki.Mapping;
using Loki.Mapping.Maps;
using Microsoft.ML.Probabilistic.Tests;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.MPFR;
using System.Text;
using System.Threading.Tasks;
using static Infer.Loki.Mappings.AssertionMethods;

namespace Infer.Loki.Tests
{
    class AssertionMap : IMap
    {
        public void MapAll(Mappers mappers)
        {
            mappers.MethodMapper.CreateMap<Func<double, double, bool>, Func<BigFloat, BigFloat, bool>>(SpecialFunctionsTests.IsErrorSignificant, IsErrorSignificantPreciser);
        }
    }
}
