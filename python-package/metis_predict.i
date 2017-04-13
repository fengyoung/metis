%module(docstring="This is the metis predict module") metis_predict

%{
    #include "../src/metis_uti.h"
    #include "../src/metis_nn.h"
    #include "../src/metis_pred.h"
    #include "../src/utility/StringArray.h"
    #include "../src/utility/Random.h"
    #include "../src/neural_network/Matrix.h"
    #include "../src/neural_network/TypeDefs.h"
    #include "../src/neural_network/Activation.h"
    #include "../src/prediction/Model.h"
    #include "../src/prediction/Model_Perceptron.h"
    #include "../src/prediction/Model_FM.h"
    #include "../src/prediction/Model_MLP.h"
    #include "../src/prediction/Model_FMSNN.h"
%}

%include <stdint.i>
%include <std_pair.i>
%include <std_string.i>
%include <std_vector.i>
%include <std_iostream.i>
%include <std_sstream.i>

%template() std::pair<int32_t, double>;
%template(PairVector) std::vector<std::pair<int32_t, double> >;

%feature("autodoc", "3");

%include "../src/metis_uti.h"
%include "../src/metis_nn.h"
%include "../src/metis_pred.h"
%include "../src/utility/StringArray.h"
%include "../src/utility/Random.h"
%include "../src/neural_network/Matrix.h"
%include "../src/neural_network/TypeDefs.h"
%include "../src/neural_network/Activation.h"
%include "../src/prediction/Model.h"
%include "../src/prediction/Model_Perceptron.h"
%include "../src/prediction/Model_FM.h"
%include "../src/prediction/Model_MLP.h"
%include "../src/prediction/Model_FMSNN.h"
