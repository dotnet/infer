using Loki.Shared;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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
    }
}
