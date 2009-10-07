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
*    OneErrorMeasure.java
*    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
*
*/

package mulan.evaluation.measure;

import mulan.classifier.MultiLabelOutput;

/**
 * Implementation of one-error measure. The computed value of one-error 
 * is from {0,1} set. The one-error is '1' if a top ranked label in the 
 * is not one of truly relevant labels. 
 * 
 * @author Jozef Vilcek
 */
public class OneErrorMeasure implements Measure {

	/** The name of the measure. */
	private final static String NAME = "OneError";
	/** The number of examples processed by 'compute' method so far. */
	private int processedExamples;
	/** The cumulated measure value. */
	private double measurureSum;

	/**
	 * {@inheritDoc}<br/>
	 * The computed value of one-error is from {0,1} set. The one-error is '1' 
	 * if a top ranked label in the is not one of truly relevant labels.
	 */
	@Override
	public double compute(MultiLabelOutput output, boolean[] trueLabels) {
		
		double oneError = 0;
        int[] ranks = output.getRanking();
		int numLabels = trueLabels.length;
        for (int topRated=0; topRated<numLabels; topRated++){
            if (ranks[topRated] == 1){
            	if (!trueLabels[topRated]){
                	oneError++;
                	measurureSum += oneError;
            	}
                break;
            }
        }
        processedExamples++;
		return oneError;
	}

	@Override
	public double getValue() {
		return measurureSum / processedExamples;
	}

	@Override
	public void reset() {
		processedExamples = 0;
		measurureSum = 0;
	}

	@Override
	public String getName() {
		return NAME;
	}

	@Override
	public double getIdealValue() {
		return 0;
	}

}
