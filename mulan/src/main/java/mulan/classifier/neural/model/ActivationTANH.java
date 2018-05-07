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
 *    ActivationTANH.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.classifier.neural.model;

/**
 * Implements the hyperbolic tangent activation function.
 * The function output values are from interval &lt;-1,1&gt;.
 * 
 * @author Jozef Vilcek
 * @version 2012.02.27
 */
public class ActivationTANH extends ActivationFunction {

    private static final long serialVersionUID = -8707244320811304601L;
    /** Maximum value of function */
    public final static double MAX = +1.0;
    /** Minimum value of function */
    public final static double MIN = -1.0;

    public double activate(final double input) {
        return 2.0 / (1.0 + Math.exp(-2.0 * input)) - 1.0;
    }

    public double derivative(final double input) {
        return 1.0 - Math.pow(activate(input), 2.0);
    }

    public double getMax() {
        return MAX;
    }

    public double getMin() {
        return MIN;
    }
}