// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

//#define writeToConsole

using System;
using System.Collections.Generic;
using System.Globalization;
using Microsoft.ML.Probabilistic.Collections;
using Microsoft.ML.Probabilistic.Compiler.CodeModel;
using Microsoft.ML.Probabilistic.Compiler.Graphs;
using Microsoft.ML.Probabilistic.Factors;
using System.Linq;
using Microsoft.ML.Probabilistic.Models.Attributes;

namespace Microsoft.ML.Probabilistic.Compiler.Transforms
{
#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning disable 1591
#endif

    /// <summary>
    /// The group transform deals with all aspects of group handling
    /// It creates groups as necessary for deterministic factors and constraints
    /// and changes the group annotations into MessagePath annotations on factor expressions.
    /// It also creates related annotations (ChannelPath attributes) on variable declarations
    /// so that the message transform knows what type of variables to create.
    /// </summary>
    /// <remarks>
    /// The group transform has the following steps:
    /// 
    /// 1. Builds a bipartite factor graph of factors and variables. This is done using the
    ///    transform framework. This will also create groups for deterministic factors and
    ///    constraints
    ///
    /// 2. For each existing group (i.e. manually specified group) consisting of more than one
    ///    variable expression, picks a variable, and finds the shortest path from that variable
    ///    to all other variables in the group. Add all nodes in the path to the group.
    ///    
    /// 3a. For any deterministic factor which is not a variable factor, creates a group
    ///     for each input argument, and add the output argument (unless such a pair exists)
    /// 3b. Similarly for constraints, where the first argument to the constraint acts as
    ///     the output argument.
    ///     
    /// 4. For each variable factor, if any argument is in a group, ensures that all other arguments
    ///     are in the group.
    ///     
    /// 5. Finds the root of each group. With the current architecture, the root variable cannot
    ///    be on the 'uses' side of any other variable in the group. This is because 'uses' are
    ///    currently treated as a whole in terms of type, and if one of the uses (or anything downstream)
    ///    were picked as the root, then one of the uses would need to be a different type.
    ///    The exception to this is GateExitRandom which is used for variables defined within
    ///    a gate - in this case, the root must be the exit side. 
    ///    
    ///       NOTE: In future, it would be nice to have a factor which allowed one of the uses to be
    ///       treated differently from all the other uses. This factor could then be inserted in situations
    ///       where roots were manually specified the uses side of any variable in the group. This would
    ///       require that the Group transform should go before the Channel transform so that we can
    ///       insert the correct type of variable factor
    ///    
    /// 6. Calculates distances from the root for each group
    ///    
    /// 7. Loops through the factor expressions, and call the algorithm to set the paths.
    ///    for the MessagePath attributes. This is done whether or not the expression
    ///    participates in a group.
    ///
    /// 8. Traverses the factor graph to create the ChannelPath attributes
    /// </remarks>
    internal class GroupTransform : ShallowCopyTransform
    {
        public override string Name
        {
            get { return "GroupTransform"; }
        }

        // The algorithm instance 
        protected IAlgorithm algorithm;

        private VariableGroup currentGroup = null;
        private GroupNode currentRoot = null;
        private IVariableDeclaration currentConditionVar;

        // The factor graph
        private Graph<GroupNode> factorGraph = new Graph<GroupNode>();

        // A dictionary of nodes in the factor graph keyed by variable declaration
        private Dictionary<IVariableDeclaration, GroupNode> nodeOfVariable = new Dictionary<IVariableDeclaration, GroupNode>();

        // List of factor expression nodes
        private List<GroupNode> factorExpressionNodes = new List<GroupNode>();

        // Dictionary mapping variable group to group member root node
        private Dictionary<VariableGroup, GroupMemberWithDistanceCount> rootOfGroup = new Dictionary<VariableGroup, GroupMemberWithDistanceCount>();

        // Dictionary mapping variable group to its GroupNodes
        private Dictionary<VariableGroup, ICollection<GroupNode>> nodesInGroup = new Dictionary<VariableGroup, ICollection<GroupNode>>();

        // The set of automatically generated groups
        private Set<VariableGroup> generatedGroups = new Set<VariableGroup>();

        /// <summary>
        /// Constucts a Group transform
        /// </summary>
        /// <param name="algorithm"></param>
        public GroupTransform(IAlgorithm algorithm)
            : base()
        {
            this.algorithm = algorithm;
        }

        protected override void DoConvertMethodBody(IList<IStatement> outputs, IList<IStatement> inputs)
        {
            base.DoConvertMethodBody(outputs, inputs);
            if (context.Results.IsErrors()) return;
            // Convert variable reference dependencies into statement dependencies.
            PostProcessDependencies();
        }

        // Create any missing groups for deterministic factors
        private void CreateMissingGroups(GroupNode gn)
        {
            if (!gn.IsFactorNode) return;
            if (gn.isVariableFactor) return;
            if (!gn.isDeterministic) return;

            IVariableDeclaration anchorDecl = gn.anchorNode.variableDeclaration;
            int parentIndex = 0;
            foreach (GroupNode vgn in gn.Neighbors)
            {
                if (object.ReferenceEquals(vgn, gn.anchorNode))
                    continue;
                parentIndex++;
                if (gn.isGateExit && parentIndex != 2) continue;
                IList<GroupMember> gms = context.InputAttributes.GetAll<GroupMember>(vgn.variableDeclaration);
                // See if there is a common group between this node and the anchor node.
                // If not, add a group member                    
                if (findGroupMatch(anchorDecl, vgn.variableDeclaration) == null)
                {
                    VariableGroup vg = new VariableGroup();
                    vg.Name = gn.factorExpression.Method.Method.Name + groupCount;
                    groupCount++;
                    generatedGroups.Add(vg);
                    nodesInGroup.Add(vg, new Set<GroupNode>());
                    nodesInGroup[vg].Add(gn.anchorNode);
                    context.OutputAttributes.Add(anchorDecl, new GroupMember(vg, false));
#if writeToConsole
                    Console.WriteLine("Creating group "+vg+" for ("+anchorDecl.Name+","+vgn.variableDeclaration.Name+")");
#endif
                    AddOrMergeGroups(vgn, vg);
                }
            }
        }

        /// <summary>
        /// Used for assigning unique names to groups
        /// </summary>
        private int groupCount;

        // Attach distance counters.
        private void AttachGroupInfo(GroupNode vgn)
        {
            IList<GroupMember> gms = context.InputAttributes.GetAll<GroupMember>(vgn.variableDeclaration);
            vgn.AttachGroupMembers(gms);
        }

        /// <summary>
        /// Mark distance from current root
        /// </summary>
        /// <param name="edge"></param>
        private void MarkDistanceFromRoot(Edge<GroupNode> edge)
        {
            GroupNode sgn = edge.Source;
            GroupNode tgn = edge.Target;
#if writeToConsole
            Console.WriteLine(sgn+" ("+sgn.currentDistance+") -> "+tgn);
#endif
            tgn.currentDistance = sgn.currentDistance + 1;
        }

        /// <summary>
        /// Attach distances for current group
        /// </summary>
        /// <param name="gn"></param>
        private void AttachDistances(GroupNode gn)
        {
            if (gn.IsFactorNode)
                return;

            foreach (GroupMemberWithDistanceCount tgm in gn.groupMembers)
            {
                if (tgm.group.Equals(currentGroup))
                {
                    tgm.distance = gn.currentDistance;
#if writeToConsole
                    Console.WriteLine("Distance of {0} from root: {1})", gn.variableDeclaration, tgm.distance);
#endif
                }
            }
        }

        // Create message path attributes from a factor to a variable. The paths are filled in by
        // the algorithm.
        private void CreateMessagePathAttr(GroupNode fgn, GroupNode vgn)
        {
            if (vgn.groupMembers == null) return;
            foreach (GroupMemberWithDistanceCount vgm in vgn.groupMembers)
            {
                string toName = fgn.argumentMap[vgn];
                foreach (GroupNode sgn in fgn.Neighbors)
                {
                    string fromName = fgn.argumentMap[sgn];
                    if (toName == fromName)
                        continue;
                    GroupMemberWithDistanceCount sgm = null;
                    foreach (GroupMemberWithDistanceCount gmwdc in sgn.groupMembers)
                        if (gmwdc.group == vgm.group)
                        {
                            sgm = gmwdc;
                            break;
                        }

                    if (sgm == null)
                        continue;
                    context.OutputAttributes.Add(fgn.factorExpression, new MessagePathAttribute(fromName, toName, sgm.distance, vgm.distance));
                }
            }
        }

        // Create channel path attributes for a given variable to factor. These are all the
        // messages paths (if any) which are associated with a given channel, and whether
        // backwards or forwards
        // Note: we never need to attach CurrentSample ChannelPaths, since that is the default
        private void CreateChannelPaths(GroupNode vgn)
        {
            if (vgn.IsFactorNode)
                return;

            Set<ChannelPathAttribute> cpas = Set<ChannelPathAttribute>.FromEnumerable(context.OutputAttributes.GetAll<ChannelPathAttribute>(vgn.variableDeclaration));

            // Loop through all attached factors
            int factorCount = 0;
            foreach (GroupNode fgn in vgn.Neighbors)
            {
                if (fgn.isVariableFactor) continue;
                factorCount++;
                // Get the factor node message path attributes
                var mpas = context.OutputAttributes.GetAll<MessagePathAttribute>(fgn.factorExpression);

                bool isOutput = factorGraph.ContainsEdge(fgn, vgn);
                bool hasStochasticSource = false;

                // We are going from this variable to the factor
                string fromName = fgn.argumentMap[vgn];
                // Loop through all the variables attached to this factor
                foreach (GroupNode sgn in fgn.Neighbors)
                {
                    if (sgn == vgn && fgn.Neighbors.Count != 1) continue;
                    string toName = fgn.argumentMap[sgn];
                    // Look for all message paths which match the from/to
                    foreach (MessagePathAttribute mpa in mpas)
                    {
                        if (mpa.AppliesTo(fromName, toName))
                        {
                            ChannelPathAttribute cpa = new ChannelPathAttribute(mpa.Path, isOutput ? MessageDirection.Backwards : MessageDirection.Forwards, mpa.IsDefault);
                            if (!cpas.Contains(cpa))
                            {
                                cpas.Add(cpa);
                                context.OutputAttributes.Add(vgn.variableDeclaration, cpa);
                            }
                        }
                        else if (mpa.AppliesTo(toName, fromName))
                        {
                            bool ignoreSource = fgn.isGateExit && IsFirstParent(fgn, sgn);
                            if (mpa.Path == "Distribution" && !ignoreSource) hasStochasticSource = true;
                        }
                    }
                }
                if (isOutput && fgn.isDeterministic && !hasStochasticSource)
                {
                    // a deterministic factor with all deterministic inputs will send a sample to the output variable
                }
                else if ((fgn.isGateExit && fromName == "values" && !hasStochasticSource) ||
                         (fgn.isCopy && fromName == "value" && !hasStochasticSource))
                {
                    // Gate.ExitRandom sends a sample to values
                    // Factor.Copy sends a sample to value
                }
                else
                {
                    // all other factors will send a distribution to the variable
                    ChannelPathAttribute cpa = new ChannelPathAttribute("Distribution", isOutput ? MessageDirection.Forwards : MessageDirection.Backwards, false);
                    if (!cpas.Contains(cpa))
                    {
                        cpas.Add(cpa);
                        context.OutputAttributes.Add(vgn.variableDeclaration, cpa);
                    }
                }
            }
            if (factorCount == 0)
            {
                // vgn must be a marginal channel or an unused uses channel
                ChannelPathAttribute cpa = new ChannelPathAttribute("Distribution", MessageDirection.Backwards, false);
                context.OutputAttributes.Add(vgn.variableDeclaration, cpa);
            }
        }

        // Check if a node is in a group. If no, add it, and add it to the list of nodes
        // for that variable group
        internal void checkAndAddToGroup(GroupNode gn, VariableGroup vg)
        {
            // See if the node is already in the group
            bool found = false;
            IList<GroupMember> gms = context.InputAttributes.GetAll<GroupMember>(gn.variableDeclaration);
            foreach (GroupMember gm in gms)
            {
                if (object.ReferenceEquals(gm.Group, vg))
                {
                    found = true;
                    break;
                }
            }
            // Not in the group so add it
            if (!found)
            {
                GroupMember gm = new GroupMember(vg, false);
                context.OutputAttributes.Add(gn.variableDeclaration, gm);
                nodesInGroup[vg].Add(gn);
            }
        }

        private void AddOrMergeGroups(GroupNode gn, VariableGroup vg)
        {
            // does the node already belong to any groups?
            VariableGroup existingGroup = null;
            IList<GroupMember> gms = context.InputAttributes.GetAll<GroupMember>(gn.variableDeclaration);
            foreach (GroupMember gm in gms)
            {
                existingGroup = gm.Group;
                break;
            }
            if (existingGroup == vg) return; // already in this group
            context.OutputAttributes.Add(gn.variableDeclaration, new GroupMember(vg, false));
            nodesInGroup[vg].Add(gn);
            if (existingGroup != null)
            {
                // merge groups
                if (generatedGroups.Contains(vg))
                {
                    MergeGroups(vg, existingGroup);
                }
                else if (generatedGroups.Contains(existingGroup))
                {
                    MergeGroups(existingGroup, vg);
                }
                else
                {
                    Error("Groups " + vg + " and " + existingGroup + " have an invalid overlap.  Try merging them into one group.");
                }
            }
        }

        private void MergeGroups(VariableGroup child, VariableGroup parent)
        {
            if (child == parent) return;
#if writeToConsole
            Console.WriteLine("Merging group "+child+" into group "+parent);
#endif
            ICollection<GroupNode> childNodes = nodesInGroup[child];
            ICollection<GroupNode> parentNodes = nodesInGroup[parent];
            // move all childNodes to the parent group
            foreach (GroupNode gn in childNodes)
            {
                bool alreadyInParentGroup = parentNodes.Contains(gn);
                if (!alreadyInParentGroup) parentNodes.Add(gn);
                if (gn.variableDeclaration == null) continue;
                if (!alreadyInParentGroup) context.InputAttributes.Add(gn.variableDeclaration, new GroupMember(parent, false));
                var newGroups = context.InputAttributes.GetAll<GroupMember>(gn.variableDeclaration)
                    .Where(gm => gm.Group != child);
                context.InputAttributes.Remove<GroupMember>(gn.variableDeclaration);
                foreach (GroupMember gm in newGroups) context.InputAttributes.Add(gn.variableDeclaration, gm);
            }
            nodesInGroup.Remove(child);
        }

        private IEnumerable<GroupNode> ChildDetFactorsOf(GroupNode gn)
        {
            foreach (GroupNode childNode in gn.Targets)
            {
                foreach (GroupNode gn2 in childNode.Neighbors)
                {
                    if (gn2 == gn) continue;
                    if (!gn2.IsFactorNode || !(gn2.isDeterministic || gn2.isGateExit || gn2.isVariableFactor) || gn2.Targets.Count == 0) continue;
                    yield return gn2;
                }
            }
        }

        private bool IsFirstParent(GroupNode fn, GroupNode vn)
        {
            foreach (GroupNode tn in fn.Sources)
            {
                return (tn == vn);
            }
            return false;
        }

        private IEnumerable<GroupNode> TargetsInGroup(ICollection<GroupNode> nodes, GroupNode gn)
        {
            foreach (GroupNode gn2 in gn.Targets)
            {
                if (gn2.IsFactorNode || nodes.Contains(gn2)) yield return gn2;
            }
        }

        private IEnumerable<GroupNode> NeighborsInGroup(ICollection<GroupNode> nodes, GroupNode gn)
        {
            foreach (GroupNode gn2 in gn.Neighbors)
            {
                if (gn2.IsFactorNode || nodes.Contains(gn2)) yield return gn2;
            }
        }

        private IEnumerable<GroupNode> NeighborsInSameGate(GroupNode gn)
        {
            if (gn.conditionVar != null)
            {
                foreach (GroupNode gn2 in gn.Neighbors)
                {
                    if (gn2.conditionVar == gn.conditionVar) yield return gn2;
                }
            }
        }

        private bool IsDefinitionChannel(IVariableDeclaration ivd)
        {
            ChannelInfo ci = context.InputAttributes.Get<ChannelInfo>(ivd);
            return ci != null && ci.IsDef;
        }

        private void WriteGroups(GroupNode gn)
        {
            Console.Write(gn + " groups: ");
            IList<GroupMember> gms = context.InputAttributes.GetAll<GroupMember>(gn.variableDeclaration);
            foreach (GroupMember gm in gms)
            {
                Console.Write(gm.Group);
                Console.Write(" ");
            }
            Console.WriteLine();
        }

        /// <summary>
        /// Post-process dependencies. Converts GroupMember attributes to MessagePath attributes
        /// </summary>
        internal void PostProcessDependencies()
        {
            BreadthFirstSearch<GroupNode> bfs;

#if writeToConsole
            if (false) {
                foreach (GroupNode gn in factorGraph.Nodes) {
                    foreach (GroupNode target in factorGraph.TargetsOf(gn)) {
                        Console.WriteLine(" {0} -> {1}", gn, target);
                    }
                }
                Console.WriteLine();
            }
#endif

            if (true)
            {
                // Check that each group is connected
                ICollection<GroupNode> nodes = null;
                bfs = new BreadthFirstSearch<GroupNode>(gn => NeighborsInGroup(nodes, gn), factorGraph);
                bfs.DiscoverNode += delegate(GroupNode node) { node.currentDistance = 0; };
                foreach (KeyValuePair<VariableGroup, ICollection<GroupNode>> kvp in nodesInGroup)
                {
                    // The variable group
                    VariableGroup vg = kvp.Key;
                    // All the nodes in the factor graph which are currently in this group
                    List<GroupNode> groupNodes = new List<GroupNode>(kvp.Value);
                    // If only 1, then no need to do anything
                    if (groupNodes.Count <= 1)
                        continue;
                    nodes = kvp.Value;
                    // Reset distances
                    foreach (GroupNode gn in nodes)
                    {
                        gn.currentDistance = -1;
                    }

                    bfs.Clear();
                    GroupNode start = groupNodes[0];
                    bfs.SearchFrom(start);

                    // Check that all nodes in group have nonnegative distance
                    foreach (GroupNode gn in nodes)
                    {
                        if (gn.currentDistance < 0)
                        {
                            Error("Group " + vg + " is not connected (could not reach " + gn + " from " + start + ")");
                            return;
                        }
                    }
                }
            }
            if (true)
            {
                // Check that there is no path from a stochastic var to another stochastic var in the same gate
                GroupNode stocVarNode = null;
                var dfsStoc = new DepthFirstSearch<GroupNode>(NeighborsInSameGate, factorGraph);
                dfsStoc.DiscoverNode += delegate(GroupNode node)
                    {
                        if (node != stocVarNode && !node.IsFactorNode && !context.InputAttributes.Has<DerivedVariable>(node.variableDeclaration) &&
                            IsDefinitionChannel(node.variableDeclaration))
                        {
                            Error("Gibbs Sampling does not support two random variables '" + node.variableDeclaration.Name + "' and '" + stocVarNode.variableDeclaration.Name +
                                  "' in the same condition block");
                        }
                    };
                foreach (GroupNode node in factorGraph.Nodes)
                {
                    if (node.IsFactorNode || context.InputAttributes.Has<DerivedVariable>(node.variableDeclaration) || !IsDefinitionChannel(node.variableDeclaration))
                        continue;
                    stocVarNode = node;
                    dfsStoc.SearchFrom(node);
                }
            }

            // Create groups for equality constraints
            foreach (GroupNode gn in factorGraph.Nodes)
            {
                if (!gn.isConstrainEqual) continue;
                CreateMissingGroups(gn);
            }

            //-----------------------------------------------------
            // Mark all deterministic factors with groups if they
            // are not manually marked
            //-----------------------------------------------------
            DepthFirstSearch<GroupNode> dfs = new DepthFirstSearch<GroupNode>(ChildDetFactorsOf, factorGraph);
            // an exit variable may be the child of multiple factors.  we want to group with the firstParent of only one of them.
            Set<GroupNode> variablesChecked = new Set<GroupNode>();
            dfs.FinishNode += delegate(GroupNode gn)
                {
                    if (gn.isDeterministic || gn.isVariableFactor)
                    {
                        Set<GroupNode> parents = new Set<GroupNode>();
                        int ignoreCount = 0;
                        // for GateExit we want to ignore the first argument
                        if (gn.isGateExit) ignoreCount = 1;
                        GroupNode firstParent = null;
                        foreach (GroupNode vgn in gn.Sources)
                        {
                            if (ignoreCount > 0)
                            {
                                ignoreCount--;
                                continue;
                            }
                            parents.Add(vgn);
                            if (firstParent == null) firstParent = vgn;
                        }
                        foreach (GroupNode childNode in gn.Targets)
                        {
                            if (firstParent != null && !variablesChecked.Contains(childNode))
                            {
                                // check all groups of the child variable
                                List<GroupMember> gms = context.InputAttributes.GetAll<GroupMember>(childNode.variableDeclaration);
                                foreach (GroupMember gm in gms)
                                {
                                    ICollection<GroupNode> nodes = nodesInGroup[gm.Group];
                                    // does the group contain any parent?
                                    bool containsAnyParent = nodes.ContainsAny(parents);
                                    if (!containsAnyParent)
                                    {
                                        // add the first parent to the group
#if writeToConsole
                                    Console.WriteLine("adding firstParent "+firstParent+" of "+childNode+" to group "+gm.Group);
#endif
                                        AddOrMergeGroups(firstParent, gm.Group);
                                    }
                                }
                                variablesChecked.Add(childNode);
                            }
                        }
                    }
                    CreateMissingGroups(gn);
                };
            foreach (GroupNode fgn in factorExpressionNodes)
            {
                if (!(fgn.isDeterministic || fgn.isGateExit || fgn.isVariableFactor) || fgn.Targets.Count == 0) continue;
                dfs.SearchFrom(fgn);
            }

            //--------------------------------------------
            // Attach the distance counters to the nodes
            //--------------------------------------------
            foreach (GroupNode vgn in factorGraph.Nodes)
                if (!vgn.IsFactorNode)
                    AttachGroupInfo(vgn);

            //-----------------------------------------------------
            // For each factor with a child node, mark any child variable node
            // group members as 'cannotBeRoot' if there is a parent
            // variable node within the same group (this is overridden if
            // a group has been manually set)
            //-----------------------------------------------------
            foreach (GroupNode fgn in factorExpressionNodes)
            {
                foreach (GroupNode childNode in fgn.Targets)
                {
                    foreach (GroupMemberWithDistanceCount gmdc in childNode.groupMembers)
                    {
                        VariableGroup vg = gmdc.group;
                        ICollection<GroupNode> nodes = nodesInGroup[vg];
                        // root must be a parent
                        // Is there a parent variable in the same group?
                        bool foundParent = false;
                        foreach (GroupNode vgn in fgn.Sources)
                        {
                            if (nodes.Contains(vgn))
                            {
                                foundParent = true;
                                break;
                            }
                        }
                        if (foundParent)
                        {
                            // If there is a parent in the same group, then the child cannot be a root
                            gmdc.cannotBeRoot = true;
                        }
                    }
                }
            }

            //---------------------------------------------------
            // Set up the group dictionary to point to the roots
            //---------------------------------------------------
            int grpCount = 0;
            int numGroups = nodesInGroup.Count;
            rootOfGroup.Clear();
            foreach (GroupNode gn in factorGraph.Nodes)
            {
                if (gn.IsFactorNode) continue;

                foreach (GroupMemberWithDistanceCount gmdc in gn.groupMembers)
                {
                    if (gmdc.cannotBeRoot && !gmdc.groupMember.IsRoot)
                        continue;
                    GroupMember gm = gmdc.groupMember;
                    VariableGroup vg = gm.Group;
                    bool previouslyAssigned = rootOfGroup.ContainsKey(vg);
                    // Manually assigned root overrides 'cannotBeRoot'
                    if ((!previouslyAssigned) || gmdc.groupMember.IsRoot)
                    {
                        if (previouslyAssigned)
                        {
#if writeToConsole
                            Console.WriteLine("Undo {0} as root of {1}", rootOfGroup[vg].groupNode, vg);
#endif
                            rootOfGroup[vg].groupMember.IsRoot = false;
                            rootOfGroup.Remove(vg);
                        }
                        else
                        {
                            grpCount++;
                        }
                        IVariableDeclaration ivd = gmdc.groupNode.variableDeclaration;
                        if (context.InputAttributes.GetAll<GroupMember>(ivd).Count > 1)
                            Error("Variable '" + ivd.Name + "' belongs to multiple groups so it cannot be a root");
#if writeToConsole
                        Console.WriteLine("{0} is root of {1}", gn, vg);
#endif
                        gm.IsRoot = true;
                        rootOfGroup.Add(vg, gmdc);
                    }
                    if (grpCount >= numGroups)
                        break;
                }
            }

            if (rootOfGroup.Count != nodesInGroup.Count)
                Error("Cannot find root for " + (nodesInGroup.Count - rootOfGroup.Count).ToString(CultureInfo.InvariantCulture) + " groups");

            //--------------------------------------------
            // Mark distances for each variable group
            //--------------------------------------------
            bool distanceMethod1 = true;
            if (distanceMethod1)
            {
                ICollection<GroupNode> nodes = null;
                bfs = new BreadthFirstSearch<GroupNode>(gn => NeighborsInGroup(nodes, gn), factorGraph);
                bfs.TreeEdge += MarkDistanceFromRoot;
                bfs.FinishNode += AttachDistances;
                foreach (KeyValuePair<VariableGroup, GroupMemberWithDistanceCount> kvp in rootOfGroup)
                {
                    currentGroup = kvp.Key;
                    nodes = nodesInGroup[currentGroup];
                    foreach (GroupNode gn in nodes)
                        gn.currentDistance = -1;
                    currentRoot = kvp.Value.groupNode;
                    currentRoot.currentDistance = 0;

                    // Find the distances from the root
#if writeToConsole
                    Console.WriteLine("\n\nGroup {0}. From root ({1})", currentGroup, currentRoot.variableDeclaration);
#endif
                    bfs.Clear();
                    bfs.SearchFrom(currentRoot);
                }
            }
            else
            {
                // TODO: change above to use this method of computing distances
                foreach (KeyValuePair<VariableGroup, GroupMemberWithDistanceCount> kvp in rootOfGroup)
                {
                    currentGroup = kvp.Key;
                    currentRoot = kvp.Value.groupNode;

                    // Attach the distances to the GroupMembers
#if writeToConsole
                    Console.WriteLine("\n\nGroup {0}. From root ({1})", currentGroup, currentRoot.variableDeclaration);
#endif
                    // Find the distances from the root
                    DistanceSearch<GroupNode> distSearch = new DistanceSearch<GroupNode>(factorGraph);
                    //distSearch.Clear();
                    distSearch.SetDistance += delegate(GroupNode gn, int distance)
                        {
                            gn.currentDistance = distance;
                            AttachDistances(gn);
                        };
                    distSearch.SearchFrom(currentRoot);
                }
            }

            //--------------------------------------------
            // Initialise the message path attributes
            //--------------------------------------------
            foreach (GroupNode fgn in factorExpressionNodes)
            {
                if (fgn.isVariableFactor) continue;
                foreach (GroupNode vgn in factorGraph.NeighborsOf(fgn))
                    CreateMessagePathAttr(fgn, vgn);
            }

            //--------------------------------------------
            // Now let the algorithm modify message paths,
            // or create default message paths. This needs
            // to be done whether or not there are groups
            //--------------------------------------------
            foreach (GroupNode fgn in factorExpressionNodes)
                algorithm.ModifyFactorAttributes(fgn.factorExpression, context.OutputAttributes);

            //--------------------------------------------
            // Create the channel path attributes
            //--------------------------------------------
            foreach (GroupNode gn in factorGraph.Nodes)
            {
                if (gn.IsFactorNode) continue;
                CreateChannelPaths(gn);
            }
        }

        // See if two variable declarations share a common group
        private VariableGroup findGroupMatch(IVariableDeclaration ivd1, IVariableDeclaration ivd2)
        {
            IList<GroupMember> gms1 = context.InputAttributes.GetAll<GroupMember>(ivd1);
            IList<GroupMember> gms2 = context.InputAttributes.GetAll<GroupMember>(ivd2);
            foreach (GroupMember gm1 in gms1)
            {
                foreach (GroupMember gm2 in gms2)
                    if (object.ReferenceEquals(gm2.Group, gm1.Group))
                        return gm2.Group;
            }
            return null;
        }

        // See if a variable is in a particular group
        private GroupMember findGroupMatch(VariableGroup vg, IVariableDeclaration ivd)
        {
            IList<GroupMember> gms = context.InputAttributes.GetAll<GroupMember>(ivd);
            GroupMember match = null;
            foreach (GroupMember gm in gms)
            {
                if (object.ReferenceEquals(gm.Group, vg))
                {
                    match = gm;
                    break;
                }
            }
            return match;
        }

        protected override IVariableDeclaration ConvertVariableDecl(IVariableDeclaration ivd)
        {
            if (CodeRecognizer.IsStochastic(context, ivd))
            {
                GroupNode vgn;
                if (!nodeOfVariable.TryGetValue(ivd, out vgn))
                {
                    vgn = GroupNode.FromVariable(ivd);
                    nodeOfVariable[ivd] = vgn;
                    factorGraph.Nodes.Add(vgn);
                }
                vgn.conditionVar = currentConditionVar;
            }
            return base.ConvertVariableDecl(ivd);
        }

        protected override IStatement ConvertExpressionStatement(IExpressionStatement ies)
        {
            // This bit of code builds the factor graph. We are only interested in factor expressions
            // with or without a LHS

            // Is this from assignment?
            IAssignExpression iae = ies.Expression as IAssignExpression;

            // Get the target if any
            IVariableDeclaration targetVarDecl = null;
            IExpression targetExpression = null;
            if (iae != null)
            {
                targetExpression = iae.Target;
                // The following should return null if the variable is observed (as the targetExpression
                // is an argument reference expression. This situation will then be dealt with
                // as a constraint
                targetVarDecl = Recognizer.GetVariableDeclaration(targetExpression);
                // ignore constants
                if (targetVarDecl != null && !CodeRecognizer.IsStochastic(context, targetVarDecl)) targetVarDecl = null;
            }

            IMethodInvokeExpression factorExpression;
            if (iae == null)
                factorExpression = ies.Expression as IMethodInvokeExpression;
            else
                factorExpression = iae.Expression as IMethodInvokeExpression;

            FactorManager.FactorInfo info = null;

            // Factor information to get field names
            if (factorExpression != null)
                info = CodeRecognizer.GetFactorInfo(context, factorExpression);
            if (factorExpression != null && info != null && CodeRecognizer.IsStochastic(context, factorExpression))
            {
                bool isDeterministic = info.IsDeterministicFactor;
                bool isVariable = context.InputAttributes.Has<IsVariableFactor>(factorExpression);
                bool isGateExit = (info.Method.DeclaringType == typeof (Gate) && info.Method.Name.StartsWith("Exit"));
                bool isGateEnter = (info.Method.DeclaringType == typeof (Gate) && info.Method.Name.StartsWith("Enter"));
                bool isCopy = (info.Method.DeclaringType == typeof (Factor) && info.Method.Name == "Copy");
                bool isCasesCopy = context.InputAttributes.Has<CasesCopy>(factorExpression);
                bool isConstrainEqual = (info.Method.DeclaringType == typeof (Constrain) && info.Method.Name == "Equal");

                // Create and add the factor node
                GroupNode exprNode = GroupNode.FromExpression(factorExpression);
                exprNode.isDeterministic = isDeterministic;
                exprNode.isVariableFactor = isVariable;
                exprNode.isGateExit = isGateExit;
                exprNode.isGateEnter = isGateEnter;
                exprNode.isCopy = isCopy;
                exprNode.isConstrainEqual = isConstrainEqual;
                exprNode.conditionVar = currentConditionVar;
                factorGraph.Nodes.Add(exprNode);
                factorExpressionNodes.Add(exprNode);

                int fieldIndex = 0;
                // If the target variable is not already in the graph, create the node and add it
                GroupNode targetGN = null;
                if (targetVarDecl != null)
                {
                    string tgtArgName = info.ParameterNames[fieldIndex++];
                    if (!nodeOfVariable.TryGetValue(targetVarDecl, out targetGN))
                    {
                        targetGN = GroupNode.FromVariable(targetVarDecl);
                        nodeOfVariable.Add(targetVarDecl, targetGN);
                        factorGraph.Nodes.Add(targetGN);
                    }
                    // Update the factor mapping
                    exprNode.argumentMap.Add(targetGN, tgtArgName);
                    factorGraph.AddEdge(exprNode, targetGN);
                    //factorGraph.AddEdge(targetGN, exprNode);
                    IList<GroupMember> gms = context.InputAttributes.GetAll<GroupMember>(targetVarDecl);
                    foreach (GroupMember gm in gms)
                    {
                        //gm.IsRoot = false; // Do not allow manual setting of root
                        ICollection<GroupNode> nodes;
                        if (!nodesInGroup.TryGetValue(gm.Group, out nodes))
                        {
                            nodes = new Set<GroupNode>();
                            nodesInGroup[gm.Group] = nodes;
                        }
                        if (!nodes.Contains(targetGN))
                            nodes.Add(targetGN);
                    }
                    exprNode.anchorNode = targetGN;
                }

                for (int i = 0; i < factorExpression.Arguments.Count; i++)
                {
                    string srcArgName = info.ParameterNames[fieldIndex++];
                    IExpression arg = factorExpression.Arguments[i];
                    while (arg is IMethodInvokeExpression)
                    {
                        IMethodInvokeExpression imie2 = (IMethodInvokeExpression) arg;
                        if (imie2.Arguments.Count == 0) break;
                        arg = imie2.Arguments[0];
                    }
                    bool isOut = (arg is IAddressOutExpression);
                    IVariableDeclaration ivd = Recognizer.GetVariableDeclaration(arg);
                    if (ivd == null || !CodeRecognizer.IsStochastic(context, ivd))
                        continue;
                    GroupNode sourceGN;
                    if (!nodeOfVariable.TryGetValue(ivd, out sourceGN))
                    {
                        sourceGN = GroupNode.FromVariable(ivd);
                        nodeOfVariable.Add(ivd, sourceGN);
                        factorGraph.Nodes.Add(sourceGN);
                    }
                    // Update the factor mapping
                    exprNode.argumentMap.Add(sourceGN, srcArgName);
                    // Do not create an edge between a GateEnter factor and its first argument ("cases")
                    if (!(isGateEnter && i == 0))
                    {
                        if (isOut) factorGraph.AddEdge(exprNode, sourceGN);
                        else factorGraph.AddEdge(sourceGN, exprNode);
                    }

                    if (exprNode.anchorNode == null)
                        exprNode.anchorNode = sourceGN;
                    IList<GroupMember> gms = context.InputAttributes.GetAll<GroupMember>(ivd);
                    foreach (GroupMember gm in gms)
                    {
                        //gm.IsRoot = false; // Do not allow manual setting of root
                        ICollection<GroupNode> nodes;
                        if (!nodesInGroup.TryGetValue(gm.Group, out nodes))
                        {
                            nodes = new Set<GroupNode>();
                            nodesInGroup[gm.Group] = nodes;
                        }
                        if (!nodes.Contains(sourceGN))
                            nodes.Add(sourceGN);
                    }
                }
            }

            // Call the base class
            ConvertExpression(ies.Expression);
            return ies;
        }

        /// <summary>
        /// Shallow copy of for statement
        /// </summary>
        protected override IStatement ConvertFor(IForStatement ifs)
        {
            // We do not convert the loop counter declaration or increment.  We only convert
            // the loop size which is the right hand size of the loop condition.
            IBinaryExpression ibe = ifs.Condition as IBinaryExpression;
            if (ibe == null)
            {
                Error("For loop conditions must be binary expressions, was :" + ifs.Condition);
            }
            else
            {
                IVariableDeclaration loopVar = CodeRecognizer.Instance.GetVariableDeclaration(ibe.Left);
                if (loopVar == null)
                    Error("For loop conditions have loop counter reference on LHS, was :" + ibe.Left);
                if ((ibe.Right is ILiteralExpression) && 0.Equals(((ILiteralExpression) ibe.Right).Value))
                {
                    // loop of zero length
                    return null;
                }
                ConvertExpression(ibe.Right);
            }
            ConvertBlock(ifs.Body);
            return ifs;
        }

        protected override IStatement ConvertCondition(IConditionStatement ics)
        {
            ConvertExpression(ics.Condition);
            IVariableDeclaration oldConditionVar = currentConditionVar;
            if (currentConditionVar == null && CodeRecognizer.IsStochastic(context, ics.Condition))
                currentConditionVar = Recognizer.GetVariableDeclaration(ics.Condition);
            context.SetPrimaryOutput(ics);
            ConvertBlock(ics.Then);
            if (ics.Else != null) ConvertBlock(ics.Else);
            currentConditionVar = oldConditionVar;
            return ics;
        }
    }

    // Wraps group member with distant count
    internal class GroupMemberWithDistanceCount
    {
        public GroupMember groupMember; // Wrapped group member
        public GroupNode groupNode; // Owner

        public VariableGroup group
        {
            get { return groupMember.Group; }
        }

        public int distance;
        public bool cannotBeRoot = false;

        public GroupMemberWithDistanceCount(GroupNode gn, GroupMember gm)
        {
            groupNode = gn;
            groupMember = gm;
            distance = -1;
        }

        public override string ToString()
        {
            return String.Format("Distance {0})", distance);
        }
    }

    // Node in factor graph which carries group information
    internal class GroupNode : DirectedNode<GroupNode>, ICloneable
    {
        /// <summary>
        /// For a factor node, this is always null.
        /// </summary>
        public IVariableDeclaration variableDeclaration;

        /// <summary>
        /// For a node inside a gate, stores the conditionVar.  Otherwise null.
        /// </summary>
        public IVariableDeclaration conditionVar;

        /// <summary>
        /// Stores distance info for each group that this variable is a member of.
        /// </summary>
        public IList<GroupMemberWithDistanceCount> groupMembers;

        /// <summary>
        /// For a variable node, this is always null.
        /// </summary>
        public IMethodInvokeExpression factorExpression;

        public IDictionary<GroupNode, string> argumentMap;
        public bool isDeterministic;
        public bool isVariableFactor;
        public bool isGateExit;
        public bool isGateEnter;
        public bool isCopy;
        public bool isConstrainEqual;
        public GroupNode anchorNode; // Child node if factor, first arg if constraint
        // Following used for path find algorithm
        internal int currentDistance;
        internal GroupNode previousNodeInPath;

        public static GroupNode FromExpression(IMethodInvokeExpression imie)
        {
            GroupNode gn = new GroupNode();
            gn.factorExpression = imie;
            gn.variableDeclaration = null;
            gn.groupMembers = null;
            gn.argumentMap = new Dictionary<GroupNode, string>();
            return gn;
        }

        public static GroupNode FromVariable(IVariableDeclaration ivd)
        {
            GroupNode gn = new GroupNode();
            gn.variableDeclaration = ivd;
            gn.groupMembers = null;
            gn.factorExpression = null;
            gn.argumentMap = null;
            return gn;
        }

        public void AttachGroupMembers(IList<GroupMember> gms)
        {
            groupMembers = new List<GroupMemberWithDistanceCount>();
            foreach (GroupMember gm in gms)
            {
                groupMembers.Add(new GroupMemberWithDistanceCount(this, gm));
            }
        }

        public bool IsFactorNode
        {
            get { return factorExpression != null; }
        }

        public object Clone()
        {
            GroupNode that = new GroupNode();
            that.variableDeclaration = this.variableDeclaration;
            that.groupMembers = this.groupMembers;
            that.factorExpression = this.factorExpression;
            that.argumentMap = this.argumentMap;
            return that;
        }

        public override bool Equals(object obj)
        {
            GroupNode that = (GroupNode) obj;
            if (that == null)
                return false;

            if (factorExpression != null)
                return object.ReferenceEquals(factorExpression, that.factorExpression);
            else
                return object.ReferenceEquals(variableDeclaration, that.variableDeclaration);
        }

        public override int GetHashCode()
        {
            if (factorExpression != null)
                return factorExpression.GetHashCode();
            else
                return variableDeclaration.GetHashCode();
        }

        public override string ToString()
        {
            if (this.factorExpression != null)
                return "Factor (" + this.factorExpression.ToString() + ")";
            else
                return "Variable " + this.variableDeclaration.Name;
        }
    }

    /// <summary>
    /// Inherit parent grouping
    /// </summary>
    internal class InheritParentGrouping
    {
        /// <summary>
        /// ToString override
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return "Inherits parent grouping";
        }
    }

    /// <summary>
    /// Channel path attribute. Marks all paths that will exist on the channel
    /// </summary>
    internal class ChannelPathAttribute : ICompilerAttribute
    {
        /// <summary>
        ///  The message path
        /// </summary>
        public readonly string Path;

        /// <summary>
        /// Message direction
        /// </summary>
        public readonly MessageDirection Direction;

        /// <summary>
        /// Path attribute comes from a default message attribute
        /// </summary>
        public readonly bool FromDefault;

        public ChannelPathAttribute(string path, MessageDirection direction, bool fromDefault)
        {
            Path = path;
            Direction = direction;
            FromDefault = fromDefault;
        }

        /// <summary>
        /// Equals override
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object obj)
        {
            if (object.ReferenceEquals(this, obj))
                return true;
            ChannelPathAttribute cpa = obj as ChannelPathAttribute;
            if (cpa == null) return false;
            return (Path == cpa.Path && Direction == cpa.Direction && FromDefault == cpa.FromDefault);
        }

        /// <summary>
        /// ToString override
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return String.Format("ChannelPath {0}_{1}{2}",
                                 Path, (Direction == MessageDirection.Backwards) ? "B" : "F", FromDefault ? " (default)" : "");
        }

        /// <summary>
        /// GetHashCode override
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            if (Path != null)
                return Path.GetHashCode() + Direction.GetHashCode();
            else
                return Direction.GetHashCode();
        }
    }

    /// <summary>
    /// Attribute which is attached to the method invoke expression associated with an
    /// operator call. The attribute specifies a context-specific ('from' to 'to') string
    /// representation (in general a path) of the message expression (for example, the field
    /// name 'Sample' in a DistAndSample message type) 
    /// </summary>
    internal class MessagePathAttribute : ICompilerAttribute
    {
        /// <summary>
        /// The path of the 'from' message
        /// </summary>
        public string Path { get; set; }

        /// <summary>
        /// The 'from' argument for the message
        /// </summary>
        public string From { get; set; }

        /// <summary>
        /// The 'to' argument for the message
        /// </summary>
        public string To { get; set; }

        /// <summary>
        /// The distance of the 'from' argument from the root of the group
        /// </summary>
        public int FromDistance { get; set; }

        /// <summary>
        /// The distance of the 'to' argument from the root of the group
        /// </summary>
        public int ToDistance { get; set; }

        /// <summary>
        /// Indicates that the message attribute is a default attribute
        /// </summary>
        public bool IsDefault { get; set; }

        /// <summary>
        /// Create a default message path attribute
        /// </summary>
        /// <param name="path">The message path - this is a property name in the message type</param>
        public MessagePathAttribute(string path)
        {
            From = null;
            To = null;
            FromDistance = 0;
            ToDistance = 0;
            Path = path;
            IsDefault = false;
        }

        /// <summary>
        /// Create a default message path attribute associated with a variable group
        /// </summary>
        /// <param name="from">The 'from' argument name</param>
        /// <param name="to">The 'to' argument name</param>
        /// <param name="fromDistance">Distance of from argument to root</param>
        /// <param name="toDistance">Distance of to argument to root</param>
        public MessagePathAttribute(string from, string to, int fromDistance, int toDistance)
        {
            From = from;
            To = to;
            FromDistance = fromDistance;
            ToDistance = toDistance;
            Path = null; // to be set by the algorithm based on TowardsRoot
            IsDefault = false;
        }

        /// <summary>
        /// Create a message path attribute with a specified path
        /// </summary>
        /// <param name="from">The 'from' argument name</param>
        /// <param name="to">The 'to' argument name</param>
        /// <param name="path">path</param>
        /// <param name="isDefault">Whether this MPA is a default MPA</param>
        public MessagePathAttribute(string from, string to, string path, bool isDefault)
        {
            From = from;
            To = to;
            FromDistance = 0;
            ToDistance = 0;
            Path = path;
            IsDefault = isDefault;
        }

        /// <summary>
        /// Whether this attribute is applicable to specified from/to argument names
        /// </summary>
        /// <param name="from"></param>
        /// <param name="to"></param>
        /// <returns></returns>
        public bool AppliesTo(string from, string to)
        {
            if (From != null && from != null && From != from)
                return false;
            if (To != null && to != null && To != to)
                return false;

            return true;
        }

        /// <summary>
        /// ToString override
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            string fromStr = (From == null) ? "Any" : From;
            string toStr = (To == null) ? "Any" : To;
            string pathStr = String.Format("Path through factor: {0}.{1} to {2}", fromStr, (Path == null) ? "" : Path, toStr);
            string rootStr =
                (FromDistance > ToDistance)
                    ? "towards root"
                    : (FromDistance < ToDistance) ? "away from root" : "";
            return pathStr + " " + rootStr + String.Format(" ({0},{1})", FromDistance, ToDistance);
        }
    }

#if SUPPRESS_XMLDOC_WARNINGS
#pragma warning restore 1591
#endif
}