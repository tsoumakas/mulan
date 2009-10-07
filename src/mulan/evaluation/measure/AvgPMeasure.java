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

import java.util.ArrayList;
import java.util.List;

import mulan.classifier.MultiLabelOutput;

/**
 * Implementation of average precision measure. 
 * The measures measures average proportion of relevant labels in ranked list of labels.
 * The average is taken over all of the positions of the relevant labels in the ranked list.
 * 
 * @author Jozef Vilcek
 */
public class AvgPMeasure implements Measure {

	/** The name of the measure. */
	private final static String NAME = "AveragePrecision";
	/** The number of examples processed by 'compute' method so far. */
	private int processedExamples;
	/** The cumulated measure value. */
	private double measurureSum;

	@Override
	public double compute(MultiLabelOutput output, boolean[] trueLabels) {

		double avgP = 0; 
        int[] ranks = output.getRanking();
		int numLabels = trueLabels.length;
		List<Integer> relevant = new ArrayList<Integer>();
		for(int index = 0; index < numLabels; index++){
			if(trueLabels[index]){
				relevant.add(index);
			}
		}
		
		if (relevant.size() != 0){
            for (int r : relevant) {
                double rankedAbove=0;
                for (int rr : relevant){
                    if (ranks[rr] <= ranks[r]){
                        rankedAbove++;
                    }
                }
                avgP += (rankedAbove / ranks[r]);
            }
            avgP /= relevant.size();
            measurureSum += avgP;
            processedExamples++;
        }
		        
		return avgP;
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
		return 1;
	}

}
