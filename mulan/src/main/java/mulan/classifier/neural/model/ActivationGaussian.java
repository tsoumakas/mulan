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
 *    ActivationSigmoid.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.classifier.neural.model;

/**
 * Implements the gaussian activation function.
 * The function output values are from interval (0, 1).
 *
 * @author Ioannis Charitos
 * @version 2017.02.10
 */
public class ActivationGaussian extends ActivationFunction {

    private static final long serialVersionUID = 5L;
    /**
     * Maximum value of function
     */
    public final static double MAX = +1.0;
    /**
     * Minimum value of function
     */
    public final static double MIN = 0.0;

    @Override
    public double activate(double input) {
        return Math.exp(-(input * input));
    }

    @Override
    public double derivative(double input) {
        /* Due to ambiguity we manually set the derivative's value to 0.
         *  It can be proved that the function's limit
         *  at both positive and negative infinity is 0.
         */
        if (input == Double.NEGATIVE_INFINITY || input == Double.POSITIVE_INFINITY)
            return 0;
        else
            return -2.0 * input * activate(input);
    }

    @Override
    public double getMax() {
        return MAX;
    }

    @Override
    public double getMin() {
        return MIN;
    }

}
