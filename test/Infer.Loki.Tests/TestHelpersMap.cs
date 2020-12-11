using Infer.Loki.Mappings;
using Loki.Mapping;
using Loki.Mapping.Maps;
using Loki.Mapping.Methods;
using Microsoft.ML.Probabilistic.Tests;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.MPFR;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace Infer.Loki.Tests
{
    class TestHelpersMap : IMap
    {
        public void MapAll(Mappers mappers)
        {
            mappers.ManagedMethodMapper.CreateMap
            (
                typeof(SpecialFunctionsTests).GetMethod(
                    "DoubleTryParseWithWorkarounds",
                    BindingFlags.Static | BindingFlags.NonPublic,
                    null,
                    new Type[] { typeof(string), typeof(double).MakeByRefType() },
                    null),
                typeof(TestHelpersMethods).GetMethod(
                    "TryParseInvariant",
                    BindingFlags.Static | BindingFlags.Public,
                    null,
                    new Type[] { typeof(string), typeof(BigFloat).MakeByRefType() },
                    null)
            );

            mappers.DirectMemberMapper.CreateMap(
                typeof(TestUtils).GetProperty(nameof(TestUtils.DataFolderPath)),
                typeof(TestHelpersMethods).GetField(nameof(TestHelpersMethods.DataFolderPath)));
        }
    }
}
