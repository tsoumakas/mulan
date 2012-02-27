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
 *    ActivationFunction.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.classifier.neural.model;

import java.io.Serializable;

/**
 * Abstract base class for activation functions. 
 * The activation function is used in neural network to transform an input of 
 * each layer (neuron) and produce the output for next layer (neuron).
 * Depending on learning algorithm, derivation of activation function might be necessary.
 * 
 * @author Jozef Vilcek
 * @version 2012.02.27
 */
public abstract class ActivationFunction implements Serializable {

    /**
     * Computes an output value of the function for given input.
     *
     * @param input the input value to the function
     * @return the output value
     */
    public abstract double activate(final double input);

    /**
     * Computes an output value of function derivation for given input.
     *
     * @param input the input value to the function
     * @return the output value
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