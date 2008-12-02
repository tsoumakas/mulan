
package mulan.classifier.neural;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

/**
 * Implementation of a neuron.  
 * 
 * @author Jozef Vilcek
 */
public class Neuron implements Serializable {

	private static final long serialVersionUID = -2826468439369586864L;

	private double[] inputWeights;
	private double[] deltaValues; // for momentum
	private double errorValue;
	private final ActivationFunction function;
	private final double biasInput;
	private double neuronInput;
	private double neuronOutput;
	private List<Neuron> nextNeurons;
	
	// the dimension of input pattern vector without bias term
	private final int inputDim;
	
	
	/**
	 * Creates a {@link Neuron} instance.
	 * 
	 * @param function the activation function of the neuron
	 * @param inputDim the dimension of input pattern vector the neuron can process (the bias not included)
	 * @param biasValue the bias input value 
	 */
	public Neuron(final ActivationFunction function, int inputDim, double biasValue) {

		this.inputDim = inputDim;
		this.function = function;
		biasInput = biasValue;
		inputWeights = new double[inputDim+1];
		deltaValues = new double[inputDim+1];
		nextNeurons = new ArrayList<Neuron>();
		reset();
	}

	/**
	 * Creates a {@link Neuron} instance.
	 * 
	 * @param function the activation function of the neuron
	 * @param inputDim the dimension of input pattern vector the neuron can process (the bias not included)
	 * @param biasValue the bias input value 
	 * @param nextNeurons collection of neurons for which this neuron will be an input. 
	 */
	public Neuron(final ActivationFunction function, int inputDim, double biasValue, final Collection<Neuron> nextNeurons) {

		this(function, inputDim, biasValue);
		this.nextNeurons = new ArrayList<Neuron>(nextNeurons);
	}

	/**
	 * Returns activation function used by the neural layer.
	 * @return
	 */
	public ActivationFunction getActivationFunction(){
		return function;
	}
	
	/**
	 * Returns weights of the {@link Neuron}. <br/>
	 * The index of returned array corresponds to input pattern dimension + 1 for a bias.
	 * Weight for a bias is at the end of returned array. <br/> 
	 * 
	 * @return weights of the {@link Neuron} 
	 */
	public double[] getWeights() {
		return inputWeights;
	}

	/**
	 * Returns error term of the {@link Neuron}. <br/>
	 * 
	 * @return error term
	 */
	public double getError() {
		return errorValue;
	}
	
	/**
	 * Sets the error term of the {@link Neuron}. <br/>
	 */
	public void setError(double error) {
		errorValue = error;
	}

	/**
	 * Returns deltas of the {@link Neuron}. Deltas are terms, which are used 
	 * to update weights. Here are returned deltas which were computed and used 
	 * to update weights in previous learning iteration.<br/>
	 * The index of returned array corresponds to input pattern dimension + 1 for a bias. 
	 * Delta for the bias is at the end of returned array. 
	 * 
	 * @return delta values
	 */
	public double[] getDeltas() {
		return deltaValues;
	}

	/**
	 * Process an input pattern vector and returns the response of the {@link Neuron}.
	 * 
	 * @param inputs input pattern vector
	 * @return the output of the {@link Neuron}
	 */
	public double processInput(final double[] inputs) {
		
		if(inputs.length != inputDim){
			throw new IllegalArgumentException("The dimension of input pattern vector " +
					"does not match dimenstion of the neuron.");
		}
		
		neuronInput=0;
		for(int i=0; i<inputDim; i++ ){
			neuronInput += inputWeights[i]*inputs[i];
		}
		// add bias
		neuronInput += inputWeights[inputDim]*biasInput;
		neuronOutput = function.activate(neuronInput);
		
		return neuronOutput;
	}
	
	/**
	 * Returns the output of the {@link Neuron}.
	 * The output value is cached from processing of last input.
	 * 
	 * @return output of the {@link Neuron} or 0 if no 
	 * 			pattern was processed yet or layer is after reset.
	 */
	public double getOutput() {
		return neuronOutput;
	}
	
	/**
	 * Returns an input value of the {@link Neuron}. 
	 * The value is input pattern multiplied with weights and summed 
	 * across all weights of particular neuron. The output of the 
	 * neuron is then input transformed by activation function.
	 * <br/>
	 * The input values are cached from last processed input pattern.
	 * 
	 * @return the input value of the {@link Neuron} or 0 if no 
	 * 			pattern was processed yet or layer is after reset.
	 */
	public double getNeuronInput(){
		return neuronInput;
	}
	
	/**
	 * Returns a bias input value.
	 * 
	 * @return
	 */
	public double getBiasInput(){
		return biasInput;
	}
	
	/**
	 * Adds a connection to a specified {@link Neuron}.<br/> 
	 * The passed instance is assumed to be connected to the 
	 * output of this instance (forward connections only).
	 *  
	 * @param neuron the neuron which is connected to the output of this instance.
	 * @return true if specified neuron was successfully connected;
	 * 		   false if connection already exists 
	 */
	public boolean addNeuron(Neuron neuron){
		if(nextNeurons.contains(neuron)){
			return false;
		}
		return nextNeurons.add(neuron);
	}
	
	/**
	 * Adds connections to all specified {@link Neuron} instances.<br/> 
	 * Each instance of the collection is assumed to be connected to the 
	 * output of this instance (forward connections only).
	 *  
	 * @param neurons the collection of neurons which have to be connected to the output of this instance.
	 * @return true if at least one of specified neurons was successfully connected;
	 * 		   false if no connection was made. This means that all instances are already connected. 
	 */
	public boolean addAllNeurons(Collection<Neuron> neurons){
		Neuron[] items = neurons.toArray(new Neuron[0]);
		boolean nothingAdded = true;
		for(Neuron item : items){
			nothingAdded &= !this.addNeuron(item);
		}
		return !nothingAdded;
	}
	
	/**
	 * Removes a connection to a specified {@link Neuron}.<br/> 
	 * The passed instance is assumed to be connected to the 
	 * output of this instance (forward connections only).
	 *  
	 * @param neuron the neuron which is connected to the output of this instance.
	 * @return true if connection to specified neuron was successfully removed;
	 * 		   false if connection did not exist
	 */
	public boolean removeNeuron(Neuron neuron){
		return nextNeurons.remove(neuron);
	}
	
	/**
	 * Performs reset, re-initialization of the {@link Neuron}.
	 * The weights are randomly initialized, all state variables 
	 * (error term, neuron output, neuron input and deltas) are discarded. 
	 */
	public void reset(){
		
		final double max = 1.0;
		final double min = -1.0;
		final int inputsCount = inputDim + 1;
		
		errorValue = 0;
		neuronInput = 0;
		neuronOutput = 0;
		Arrays.fill(deltaValues, 0);
		for(int i=0; i<inputsCount; i++){
			inputWeights[i] = Math.random() * (max - min) + min;
		}
	}
}
