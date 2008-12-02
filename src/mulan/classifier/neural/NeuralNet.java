
package mulan.classifier.neural;

import java.util.Collections;
import java.util.List;


/**
 * Common interface for interaction with a neural network representation.
 * <br/>
 * Neural Network structure is composed of neurons organized into layers. 
 * There is one input layer, zero or more hidden layers and one output layer.
 * The input layer is used just to store and forward input pattern of the network to the first
 * hidden layer for processing. Input layer typically do not process input pattern. 
 * Neurons of input layer are assumed to have one input weight equal to 1, bias weight 
 * equal to 0 and use linear activation function.
 * 
 * @author Jozef Vilcek
 */
public interface NeuralNet {

	/**
	 * Gets the size/dimension of the input layer of the neural network. 
	 * This is the size of input pattern the neural network can process. 
	 * 
	 * @return the network input size
	 */
	int getNetInputSize();
	
	/**
	 * Gets the size/dimension of the output layer of the neural network. 
	 * This is the size of output pattern the neural network produces.
	 * 
	 * @return the network output size
	 */
	int getNetOutputSize();
	
	/**
	 * Returns a total number of layers of neural network.
	 *  
	 * @return
	 * @see NeuralNet
	 */
	int getLayersCount();

	/**
	 * Returns units of a particular layer of the neural network.
	 * The valid indexes for layers are from 0 to N-1, where N is total number of layers 
	 * <br/>
	 * The first layer (index = 0) is always input layer and 
	 * last (index = N-1) always output layer. 
	 * 
	 * @param layerIndex
	 * @return returns an unmodifiable list of units of the particular layer
	 * 		   @see Collections#unmodifiableList(List)
	 * @throws IndexOutOfBoundsException if the index is out of range
	 * @see NeuralNet
	 * @see NeuralLayer
	 */
	List<Neuron> getLayerUnits(int layerIndex);

	/**
	 * Propagates the input pattern through the network.
	 * 
	 * @param inputPattern the input pattern for the network to process
	 * @return the output of the network
	 * @throws IllegalArgumentException if input pattern is null or does not match network input dimension 
	 */
	double[] feedForward(final double[] inputPattern);
	
	/**
	 * Returns the actual output of the neural network, 
	 * which is a result of last processed input pattern.
	 * 
	 * @return the output of the network. 
	 * 			Returns null if network is reset or no input pattern was processed
	 */
	double[] getOutput();
	
	/**
	 * Perform reset, re-initialization of neural network. 
	 * All learned knowledge stored in the network will be lost. 
	 * 
	 */
	void reset();

}
