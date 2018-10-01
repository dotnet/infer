// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Probabilistic.Compiler.CodeModel;

namespace Microsoft.ML.Probabilistic.Compiler.CodeModel.Concrete
{
    /// <summary>
    /// Variable declaration
    /// </summary>
    internal class XVariableDeclaration : IVariableDeclaration
    {
        #region Fields

        private IType variableType;
        private int identifier;
        private string name;

        #endregion

        public IVariableDeclaration Variable
        {
            get { return this; }
            set { }
        }

        #region IVariableDeclaration Members

        /// <summary>
        /// Identifier for this variable
        /// </summary>
        public int Identifier
        {
            get { return this.identifier; }
            set { this.identifier = value; }
        }

        /// <summary>
        /// Name of this variable
        /// </summary>
        public string Name
        {
            get { return this.name; }
            set { this.name = value; }
        }

        /// <summary>
        /// Type of the variable
        /// </summary>
        public IType VariableType
        {
            get { return this.variableType; }
            set { this.variableType = value; }
        }

        #endregion

        #region IVariableReference Members

        /// <summary>
        /// Always returns 'this'
        /// </summary>
        /// <returns></returns>
        public IVariableDeclaration Resolve()
        {
            return this;
        }

        #endregion

        #region Object overrides

        public override string ToString()
        {
            ILanguageWriter writer = new CSharpWriter() as ILanguageWriter;
            return writer.VariableDeclarationSource(this);
        }

        public override bool Equals(object obj)
        {
            if (this == obj)
                return true;

            IVariableDeclaration declaration = obj as IVariableDeclaration;
            if (declaration == null)
                return false;

            return
                Identifier.Equals(declaration.Identifier) &&
                Name.Equals(declaration.Name);
        }

        public override int GetHashCode()
        {
            return this.Name.GetHashCode();
        }

        #endregion
    }
}