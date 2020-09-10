using Loki.Mapping.Methods;
using Loki.Shared;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics.MPFR;
using System.Text;
using System.Threading.Tasks;

namespace Infer.Loki.Mappings
{
    public static class TestHelpersMethods
    {
        public static string DataFolderPath;

        static TestHelpersMethods()
        {
            var slnDir = new DirectoryInfo(Environment.CurrentDirectory);
            while (!slnDir.EnumerateFiles().Any(fn => fn.Name == "Infer.sln"))
                slnDir = slnDir.Parent;
            DataFolderPath = Path.Combine(slnDir.FullName, "test", "Tests", "Data");
        }

        public static bool TryParseInvariant(string value, out BigFloat result) => DoubleMethods.TryParse(value, System.Globalization.NumberStyles.Float, CultureInfo.InvariantCulture, out result);
    }
}
