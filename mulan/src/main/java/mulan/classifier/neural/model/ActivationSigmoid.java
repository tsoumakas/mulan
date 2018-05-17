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
 * Implements the sigmoid activation function.
 * The function output values are from interval (0, 1).
 * 
 * @author Ioannis Charitos
 * @version 2017.01.3
 */
public class ActivationSigmoid extends ActivationFunction {
	
	
	private static final long serialVersionUID = 1L;
    /** Maximum value of function */
    public final static double MAX = + 1.0;
    /** Minimum value of function */
    public final static double MIN = 0.0;

	@Override
	public double activate(double input) {
		return 1.0 / (1.0 + Math.exp(-input)) ;
	}

	@Override
	public double derivative(double input) {
		return activate(input) * (1 - activate(input));
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
