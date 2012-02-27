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
 *    ActivationLinear.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.classifier.neural.model;

/**
 * Implements the linear activation function. The input is simply passed to the output. 
 * This activation function is commonly used for input units of networks, which serves 
 * as a place holders for input pattern and forwards them for processing. 
 * 
 * @author Jozef Vilcek
 * @version 2012.02.27
 */
public class ActivationLinear extends ActivationFunction {

    private static final long serialVersionUID = 4255801421493489832L;

    public double activate(final double input) {
        return input;
    }

    public double derivative(final double input) {
        throw new UnsupportedOperationException("Can't compute a derivative of the linear activation function.");
    }

    public double getMax() {
        return Double.POSITIVE_INFINITY;
    }

    public double getMin() {
        return Double.NEGATIVE_INFINITY;
    }
}