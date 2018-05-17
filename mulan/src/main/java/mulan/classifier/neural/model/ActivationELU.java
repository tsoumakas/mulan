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
 *    ActivationPreLU.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.classifier.neural.model;

/**
 * Implements the ELU activation function.
 * The function output values are from interval (-slope, +inf)
 * where slope is a tunable hyperparameter.
 * 
 * @author Ioannis Charitos
 * @version 2017.02.10
 */
public class ActivationELU extends ActivationFunction {
	
	private static final long serialVersionUID = 4L;
	/** Default slope value is set to 0.001 for numerical stability
	 *  in case the derivative method is called.
	 */
	private double slope = 0.001;
	/** Maximum value of function */
    public final static double MAX = Double.POSITIVE_INFINITY;
    /** Minimum value of function. 
     * It is equal to the slope times -1 thus initialized to -0.001.
     * Whenever the slope changes the MIN value must change as well */
    public static double MIN = -0.001;

	@Override
	public double activate(double input) {
		return (input >= 0) ? input : slope * (Math.exp(input) - 1);
	}

	@Override
	public double derivative(double input) {
		if (input >= 0)
			throw new UnsupportedOperationException("Can't compute a derivative "
					+ "of the ELU activation function for positve values.");
		else
			return activate(input) + slope;
	}

	@Override
	public double getMax() {
		return MAX;
	}

	@Override
	public double getMin() {
		return MIN;
	}
	
	public double getslope(){
		return slope;
	}
	
	public void setslope(double slope){
		if (slope <= 0)
			throw new IllegalArgumentException("Slope value must be greater than 0");
		else
			this.slope = slope;
			/*As the slope value changes the MIN value must do as well*/
			MIN = -slope;
	}

}
