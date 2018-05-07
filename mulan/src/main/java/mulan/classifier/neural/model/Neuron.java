/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    Neuron.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.classifier.neural.model;

import java.io.Serializable;
import java.util.*;
import mulan.core.ArgumentNullException;

/**
 * Implementation of a neuron unit. 
 * The neurons are used as processing elements in {@link NeuralNet}. 
 * 
 * @author Jozef Vilcek
 * @version 2012.02.27
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
    private final Random random;
    
    
    /**
     * Creates a new {@link Neuron} instance.
     *
     * @param function the activation function of the neuron
     * @param inputDim the dimension of input pattern vector the neuron can process (the bias not included).
     * 				   The input dimension must be greater than zero.
     * @param biasValue the bias input value
     */
    public Neuron(final ActivationFunction function, int inputDim, double biasValue) {
    	this(function, inputDim, biasValue, new Random());
    }

    /**
     * Creates a new {@link Neuron} instance.
     *
     * @param function the activation function of the neuron
     * @param inputDim the dimension of input pattern vector the neuron can process (the bias not included).
     * 				   The input dimension must be greater than zero.
     * @param biasValue the bias input value
     * @param random the pseudo-random generator to be used for computations involving randomness.
     * 		This parameter can be null. In this case, new random instance with default seed will be constructed where needed.
     */
    public Neuron(final ActivationFunction function, int inputDim, double biasValue, final Random random) {
    	if (function == null) {
            throw new ArgumentNullException("function");
        }
        if (inputDim <= 0) {
            throw new IllegalArgumentException("Input dimension for the neuron must be greather than zero.");
        }

        this.inputDim = inputDim;
        this.function = function;
        biasInput = biasValue;
        inputWeights = new double[inputDim + 1];
        deltaValues = new double[inputDim + 1];
        nextNeurons = new ArrayList<Neuron>();
        this.random = random == null ? new Random() : random;
        reset();
    }
    
    
    /**
     * Creates a new {@link Neuron} instance.
     *
     * @param function the activation function of the neuron
     * @param inputDim the dimension of input pattern vector the neuron can process (the bias not included)
     * 				   The input dimension must be greater than zero.
     * @param biasValue the bias input value
     * @param nextNeurons collection of neurons for which this neuron will be an input.
     */
    public Neuron(final ActivationFunction function, int inputDim, double biasValue, final Collection<Neuron> nextNeurons) {

        this(function, inputDim, biasValue);

        if (nextNeurons == null) {
            throw new IllegalArgumentException("Collection of connexted neurons is null.");
        }

        this.nextNeurons = new ArrayList<Neuron>(nextNeurons);
    }

    /**
     * Returns the {@link ActivationFunction} used by the {@link Neuron}.
     * @return the activation function
     */
    public ActivationFunction getActivationFunction() {
        return function;
    }

    /**
     * Returns weights of the {@link Neuron}. <br>
     * The index of returned array corresponds to input pattern dimension + 1 for a bias.
     * Weight for a bias is at the end of returned array. <br>
     *
     * @return weights of the {@link Neuron}
     */
    public double[] getWeights() {
        return inputWeights;
    }

    /**
     * Returns error term of the {@link Neuron}. <br>
     *
     * @return error term
     */
    public double getError() {
        return errorValue;
    }

    /**
     * Sets the error term of the {@link Neuron}. <br>
     *
     * @param error the error value
     */
    public void setError(double error) {
        errorValue = error;
    }

    /**
     * Returns deltas of the {@link Neuron}. Deltas are terms, which are used
     * to update weights. Here are returned deltas which were computed and used
     * to update weights in previous learning iteration.<br>
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

        if (inputs == null) {
            throw new IllegalArgumentException("The input pattern for processing is null.");
        }

        if (inputs.length != inputDim) {
            throw new IllegalArgumentException("The dimension of input pattern vector " +
                    "does not match dimenstion of the neuron.");
        }

        neuronInput = 0;
        for (int i = 0; i < inputDim; i++) {
            neuronInput += inputWeights[i] * inputs[i];
        }
        // add bias
        neuronInput += inputWeights[inputDim] * biasInput;
        neuronOutput = function.activate(neuronInput);

        return neuronOutput;
    }

    /**
     * Returns the output of the {@link Neuron}.
     * The output value is cached from processing of last input.
     *
     * @return output of the {@link Neuron} or 0 if no
     * 			pattern was processed yet or neuron is after reset.
     */
    public double getOutput() {
        return neuronOutput;
    }

    /**
     * Returns an input value of the {@link Neuron}.
     * The value is input pattern multiplied with weights and summed
     * across all weights of particular neuron. The output of the
     * neuron is then input transformed by activation function.
     * <br>
     * The input values are cached from last processed input pattern.
     *
     * @return the input value of the {@link Neuron} or 0 if no
     * 			pattern was processed yet or neuron is after reset.
     */
    public double getNeuronInput() {
        return neuronInput;
    }

    /**
     * Returns a bias input value.
     *
     * @return the input bias
     */
    public double getBiasInput() {
        return biasInput;
    }

    /**
     * Adds a connection to a specified {@link Neuron}.<br>
     * The passed instance is assumed to be connected to the
     * output of this instance (forward connections only).
     *
     * @param neuron the neuron which is connected to the output of this instance.
     * @return true if specified neuron was successfully connected;
     * 		   false if connection already exists
     * @throws IllegalArgumentException in neuron is null
     */
    public boolean addNeuron(Neuron neuron) {
        if (neuron == null) {
            throw new IllegalArgumentException("Neuron should not be null.");
        }
        if (nextNeurons.contains(neuron)) {
            return false;
        }
        return nextNeurons.add(neuron);
    }

    /**
     * Adds connections to all specified {@link Neuron} instances.<br>
     * Each instance of the collection is assumed to be connected to the
     * output of this instance (forward connections only).
     *
     * @param neurons the collection of neurons which have to be connected to the output of this instance.
     * @return true if at least one of specified neurons was successfully connected;
     * 		   false if no connection was made. This means that all instances are already connected.
     * @throws IllegalArgumentException if neurons collection is null
     */
    public boolean addAllNeurons(Collection<Neuron> neurons) {
        if (neurons == null) {
            throw new IllegalArgumentException("Neurons collection should not be null.");
        }
        Neuron[] items = neurons.toArray(new Neuron[0]);
        boolean nothingAdded = true;
        for (Neuron item : items) {
            nothingAdded &= !this.addNeuron(item);
        }
        return !nothingAdded;
    }

    /**
     * Removes a connection to a specified {@link Neuron}.<br>
     * The passed instance is assumed to be connected to the
     * output of this instance (forward connections only).
     *
     * @param neuron the neuron which is connected to the output of this instance.
     * @return true if connection to specified neuron was successfully removed;
     * 		   false if connection did not exist
     * @throws IllegalArgumentException if neuron is null
     */
    public boolean removeNeuron(Neuron neuron) {
        if (neuron == null) {
            throw new IllegalArgumentException("Neuron should not be null.");
        }
        return nextNeurons.remove(neuron);
    }

    /**
     * Performs reset, re-initialization of the {@link Neuron}.
     * The weights are randomly initialized, all state variables
     * (error term, neuron output, neuron input and deltas) are discarded.
     */
    public void reset() {

        final double max = 0.5;
        final double min = -0.5;
        final int inputsCount = inputDim + 1;

        errorValue = 0;
        neuronInput = 0;
        neuronOutput = 0;
        Arrays.fill(deltaValues, 0);
        for (int i = 0; i < inputsCount; i++) {
            inputWeights[i] = random.nextDouble() * (max - min) + min;
        }
    }

    /**
     * Gets the count of neurons connected to the output of this neuron instance.
     * Support for unit tests ...
     *
     * @return number of connected neurons
     */
    protected int getConnectedNeuronsCount() {
        return nextNeurons.size();
    }
}