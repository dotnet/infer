# Licensed to the .NET Foundation under one or more agreements.
# The .NET Foundation licenses this file to you under the MIT license.
# See the LICENSE file in the project root for more information.

if [ $# -lt 2 ]; then
    echo Usage: $0 "<output folder>" "<build configuration>" 
    exit 1
fi

# First argument is target folder. 
out=$1
# Second argument is build configuration
configuration=$2

rm -r "${out}"
mkdir "${out}"

mkdir "${out}/netstandard2.0"
dst="${out}/netstandard2.0"
src="bin/${configuration}/netstandard2.0"

cp ../src/Runtime/${src}/Microsoft.ML.Probabilistic.dll ${dst}
cp ../src/Runtime/${src}/Microsoft.ML.Probabilistic.pdb ${dst}
cp ../src/Runtime/${src}/Microsoft.ML.Probabilistic.xml ${dst}

cp ../src/Compiler/${src}/Microsoft.ML.Probabilistic.Compiler.dll ${dst}
cp ../src/Compiler/${src}/Microsoft.ML.Probabilistic.Compiler.pdb ${dst}
cp ../src/Compiler/${src}/Microsoft.ML.Probabilistic.Compiler.xml ${dst}

cp ../src/Learners/Core/${src}/Microsoft.ML.Probabilistic.Learners.dll ${dst}
cp ../src/Learners/Core/${src}/Microsoft.ML.Probabilistic.Learners.pdb ${dst}
cp ../src/Learners/Core/${src}/Microsoft.ML.Probabilistic.Learners.xml ${dst}

cp ../src/Learners/Classifier/${src}/Microsoft.ML.Probabilistic.Learners.Classifier.dll ${dst}
cp ../src/Learners/Classifier/${src}/Microsoft.ML.Probabilistic.Learners.Classifier.pdb ${dst}
cp ../src/Learners/Classifier/${src}/Microsoft.ML.Probabilistic.Learners.Classifier.xml ${dst}

cp ../src/Learners/Recommender/${src}/Microsoft.ML.Probabilistic.Learners.Recommender.dll ${dst}
cp ../src/Learners/Recommender/${src}/Microsoft.ML.Probabilistic.Learners.Recommender.pdb ${dst}
cp ../src/Learners/Recommender/${src}/Microsoft.ML.Probabilistic.Learners.Recommender.xml ${dst}

mkdir "${out}/net472"
dst="${out}/net472"
src="bin/${configuration}/net472"

cp ../src/Visualizers/Windows/${src}/Microsoft.ML.Probabilistic.Compiler.Visualizers.Windows.dll ${dst}
cp ../src/Visualizers/Windows/${src}/Microsoft.ML.Probabilistic.Compiler.Visualizers.Windows.pdb ${dst}
cp ../src/Visualizers/Windows/${src}/Microsoft.ML.Probabilistic.Compiler.Visualizers.Windows.xml ${dst}

mkdir "${out}/net6.0-windows"
dst="${out}/net6.0-windows"
src="bin/${configuration}/net6.0-windows"

cp ../src/Visualizers/Windows/${src}/Microsoft.ML.Probabilistic.Compiler.Visualizers.Windows.dll ${dst}
cp ../src/Visualizers/Windows/${src}/Microsoft.ML.Probabilistic.Compiler.Visualizers.Windows.pdb ${dst}
cp ../src/Visualizers/Windows/${src}/Microsoft.ML.Probabilistic.Compiler.Visualizers.Windows.xml ${dst}