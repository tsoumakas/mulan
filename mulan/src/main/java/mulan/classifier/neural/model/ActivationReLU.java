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
 *    ActivationReLU.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.classifier.neural.model;

/**
 * Implements the ReLU activation function.
 * The function output values are from interval (0, +inf).
 *
 * @author Ioannis Charitos
 * @version 2017.01.4
 */
public class ActivationReLU extends ActivationFunction {

    /**
     * Maximum value of function
     */
    public final static double MAX = Double.POSITIVE_INFINITY;
    /**
     * Minimum value of function
     */
    public final static double MIN = 0;
    private static final long serialVersionUID = 2L;

    @Override
    public double activate(double input) {
        return (input >= 0) ? input : 0;
    }

    @Override
    public double derivative(double input) {
        throw new
                UnsupportedOperationException("Can't compute a derivative of the ReLU activation function.");
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
