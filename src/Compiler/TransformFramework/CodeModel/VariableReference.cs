// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    internal class XVariableReference : IVariableReference
    {
        #region Fields

        internal IVariableDeclaration variable;

        #endregion

        public XVariableReference()
        {
        }

        public IVariableDeclaration Variable
        {
            get { return this.variable; }
            set { this.variable = value; }
        }

        public IVariableDeclaration Resolve()
        {
            return this.variable;
        }

        #region Object overrides

        public override string ToString()
        {
            // Base class will do the right thing
            return base.ToString();
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IVariableReference reference = obj as IVariableReference;
            if (reference == null)
                return false;

            IVariableDeclaration declThis = this.Resolve();
            IVariableDeclaration declThat = reference.Resolve();

            if (declThis == null || declThat == null)
                return false;

            return declThis.Equals(declThat);
        }

        public override int GetHashCode()
        {
            IVariableDeclaration declaration = this.Resolve();

            if (declaration == null)
                return 2147483647;
            else
                return declaration.Name.GetHashCode();
        }

        #endregion
    }
}