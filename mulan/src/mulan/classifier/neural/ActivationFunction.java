
package mulan.classifier.neural;

import java.io.Serializable;
	
/**
 * Abstract base class for activation functions. 
 * The activation function is used in neural network to transform an input of 
 * each layer (neuron) and produce the output for next layer (neuron).
 * Depending on learning algorithm, derivation of activation function might be necessary.
 * 
 * @author Jozef Vilcek
 */
public abstract class ActivationFunction implements Serializable{

	/**
	 * Computes an output value of the function for given input.
	 * 
	 * @param input the input value to the function
	 * @return
	 */
	public abstract double activate(final double input);
	
	/**
	 * Computes an output value of function derivation for given input.
	 * 
	 * @param input the input value to the function
	 * @return
	 */
	public abstract double derivative(final double input);
	
	/**
	 * Gets the maximum value the function can output.
	 * 
	 * @return maximum value of the function
	 */
	public abstract double getMax();
	
	/**
	 * Gets the minimum value the function can output.
	 * 
	 * @return minimum value of the function
	 */
	public abstract double getMin();
}
	

