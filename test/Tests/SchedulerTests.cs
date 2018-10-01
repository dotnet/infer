// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Globalization;
using System.Linq;
using Xunit;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler.Attributes;
using Microsoft.ML.Probabilistic.Compiler.Transforms;
using Microsoft.ML.Probabilistic.Compiler;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Compiler.Graphs;

namespace Microsoft.ML.Probabilistic.Tests
{
    
    public class SchedulerTests
    {
        [Fact]
        public void GroupGraphTest()
        {
            IndexedGraph g = new IndexedGraph(2);
            g.AddEdge(1, 0);
            int[] groupOf = new int[] { -1, 2, -1 };
            GroupGraph groupGraph = new GroupGraph(g, groupOf, g.Nodes.Count);
            groupGraph.BuildGroupEdges();
            var schedule = groupGraph.GetScheduleWithGroups(groupGraph.SourcesOf);
            Assert.True(schedule[0] == 1);
            Assert.True(schedule[1] == 0);
        }

        [Fact]
        public void FreshDepChainTest()
        {
            foreach (var uniform in new[] {true, false})
            {
                var schedule = RunRepair("1,0,7,6,11,7,6,5,4,3,2,1,0,11,10,9,8",
                    F(5, 0), FR(11, 0), FR(0, 1), FN(1, 2), F(2, 3), D(3, 3), F(2, 4), F(3, 4), D(2, 5), N(3, 5), R(4, 5), FR(5, 6),
                    FR(11, 6), FR(6, 7), FN(7, 8), F(8, 9), D(9, 9), F(8, 10), F(9, 10), D(5, 11), D(8, 11), N(9, 11), R(10, 11),
                    (stmts, depInfos) => { if (uniform) depInfos[8].Add(DependencyType.SkipIfUniform, new AnyStatement(stmts[8], stmts[9])); });
                Assert.Equal("1,0,7,6,11,10,9,8,7,6,5,4,3,2,1,0,11,10,9,8", schedule);
            }
        }

        [Fact]
        public void FreshDepChainTestSimple()
        {
            foreach (var uniform in new[] {true, false})
            {
                var schedule = RunRepair("0,1,0,1,2,3",
                    F(0, 3), F(3, 2), F(2, 1), F(1, 0),
                    (stmts, depInfos) => { if (uniform) depInfos[3].Add(DependencyType.SkipIfUniform, stmts[2]); });
                Assert.Equal("0,1,2,3,0,1,2,3", schedule);
            }
        }

        // Test that stale sources are not scheduled if they would be uniform
        [Fact]
        public void FreshDepChainTest2()
        {
            var schedule = RunRepair("0,0,1,2,3",
                F(0, 3), F(3, 2), U(2, 1), D(2, 0));
            Assert.Equal("0,0,1,2,3", schedule);
        }

        #region Helper methods

        private static string RunRepair(string scheduleString, params Action<IStatement[], DependencyInformation[]>[] dependencies)
        {
            var scheduleParts = scheduleString.Split(",".ToCharArray()).ToArray();
            var schedule = scheduleParts.Select(int.Parse).ToArray();

            var numStmts = schedule.Max() + 1;
            var stmts = Enumerable.Range(0, numStmts).Select(i => CodeBuilder.Instance.CommentStmt(i.ToString(CultureInfo.InvariantCulture))).ToArray();
            var context = new BasicTransformContext();
            var depInfos = stmts.Select(i => new DependencyInformation()).ToArray();
            schedule.ForEach(i => { if (!context.InputAttributes.Has<DependencyInformation>(stmts[i])) context.InputAttributes.Set(stmts[i], depInfos[i]); });
            schedule.ForEach(i => { if (!context.InputAttributes.Has<OperatorStatement>(stmts[i])) context.InputAttributes.Set(stmts[i], new OperatorStatement()); });

            dependencies.ForEach(dep => dep(stmts, depInfos));

            var dg = new DependencyGraph(context, stmts);
            var invalid = new Set<int>();
            var stale = new Set<DependencyGraph.TargetIndex>();
            var initialized = new Set<int>();
            var repaired = dg.RepairSchedule(schedule, invalid, stale, initialized);

            return string.Join(",", repaired.Select(i => i.ToString(CultureInfo.InvariantCulture)));
        }

        private static Action<IStatement[], DependencyInformation[]> D(int target, int source)
        {
            return Dependency(target, source, DependencyType.Dependency);
        }

        private static Action<IStatement[], DependencyInformation[]> F(int target, int source)
        {
            return Dependency(target, source, DependencyType.Fresh, DependencyType.Dependency);
        }

        private static Action<IStatement[], DependencyInformation[]> N(int target, int source)
        {
            return Dependency(target, source, DependencyType.NoInit, DependencyType.Dependency);
        }

        private static Action<IStatement[], DependencyInformation[]> R(int target, int source)
        {
            return Dependency(target, source, DependencyType.Requirement, DependencyType.Dependency);
        }

        private static Action<IStatement[], DependencyInformation[]> FR(int target, int source)
        {
            return Dependency(target, source, DependencyType.Fresh, DependencyType.Requirement, DependencyType.Dependency);
        }

        private static Action<IStatement[], DependencyInformation[]> FN(int target, int source)
        {
            return Dependency(target, source, DependencyType.Fresh, DependencyType.NoInit, DependencyType.Dependency);
        }

        private static Action<IStatement[], DependencyInformation[]> U(int target, int source)
        {
          return Dependency(target, source, DependencyType.SkipIfUniform, DependencyType.Dependency);
        }

        private static Action<IStatement[], DependencyInformation[]> Dependency(int target, int source, params DependencyType[] depTypes)
        {
            return (stmts, depInfos) => depTypes.ForEach(depType => depInfos[target].Add(depType, stmts[source]));
        }

        #endregion Helper methods
    }
}
