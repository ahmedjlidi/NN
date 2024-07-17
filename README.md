/*===================================================*/
/*Artifial neural network in c++ (FFN)
# Model only supports up to 2 layers currently
# Model uses relu activation function for hidden and sigmoid function for output layer
# Model only supports binary-classification currently (0 or 1)
# Model uses stochastic gradident decsent for backpropagation
# Model uses kaiming weight inisalization for weights (biases are set to 0)


@ Dataset class is used to read csv files
@ Dataset::get_as_tensor method returns 2 tensors. data and labels
@ Ann::passdata is used to pass data and labels for model
@ use model addlayer to add layers (up to 2)
@ ann::compile is used to set hyper-parameters (activation functions, learning_rate)
@Ann::info, summary, describe are used to describe the model
@ model::train trains the model, option True is set to print epochs, loss and accuracy


&{Feel free to improve the model}
====================================================*/
