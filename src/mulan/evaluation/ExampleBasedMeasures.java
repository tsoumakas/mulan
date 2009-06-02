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
 *    ExampleBasedMeasures.java
 *    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
 *
 */

package mulan.evaluation;

import mulan.classifier.MultiLabelOutput;
import weka.core.Utils;


public class ExampleBasedMeasures {
	private double hammingLoss;
	private double subsetAccuracy;
	private double accuracy;
	private double recall;
	private double precision;
	private double fMeasure;

	private double forgivenessRate;

	protected ExampleBasedMeasures(MultiLabelOutput[] output, boolean[][] trueLabels, double forgivenessRate) {
        this.forgivenessRate = forgivenessRate;
        computeMeasures(output, trueLabels);
	}
	
	protected ExampleBasedMeasures(MultiLabelOutput[] output, boolean[][] trueLabels) {
		this(output, trueLabels, 1.0);
	}

    /*
     * Aggregation of array of measures
     */
    protected ExampleBasedMeasures(ExampleBasedMeasures[] arrayOfMeasures) {
		accuracy = 0;
		hammingLoss = 0;
		precision = 0;
		recall = 0;
		fMeasure = 0;
		subsetAccuracy = 0;

        for (ExampleBasedMeasures measures : arrayOfMeasures) {
            accuracy += measures.getAccuracy();
            hammingLoss += measures.getHammingLoss();
            precision += measures.getPrecision();
            recall += measures.getRecall();
            fMeasure += measures.getFMeasure();
            subsetAccuracy += measures.getSubsetAccuracy();
        }

        int arraySize = arrayOfMeasures.length;
        accuracy /= arraySize;
        hammingLoss /= arraySize;
        precision /= arraySize;
        recall /= arraySize;
        fMeasure /= arraySize;
        subsetAccuracy /= arraySize;
    }
	
	private void computeMeasures(MultiLabelOutput[] output, boolean[][] trueLabels) {
		// Reset in case of multiple calls
		accuracy = 0;
		hammingLoss = 0;
		precision = 0;
		recall = 0;
		fMeasure = 0;
		subsetAccuracy = 0;

		int numLabels = trueLabels[0].length;
        int numInstances = output.length;
        for (int instanceIndex=0; instanceIndex<numInstances; instanceIndex++)
		{
			// Counter variables
			double setUnion = 0; // |Y or Z|
			double setIntersection = 0; // |Y and Z|
			double labelPredicted = 0; // |Z|
			double labelActual = 0; // |Y|
			double symmetricDifference = 0; // |Y xor Z|
			boolean setsIdentical = true; // innocent until proven guilty

			//Do the counting
            boolean[] bipartition = output[instanceIndex].getBipartition();
			for (int labelIndex = 0; labelIndex < numLabels; labelIndex++)
			{
				boolean actual = trueLabels[instanceIndex][labelIndex];
				boolean predicted = bipartition[labelIndex];

				if (predicted != actual)
				{
					symmetricDifference++;
					setsIdentical = false;
				}

				if (actual) labelActual++;
				if (predicted) labelPredicted++;

				if (predicted && actual) setIntersection++;
				if (predicted || actual) setUnion++;
			}

			if (setsIdentical) subsetAccuracy++;

			if(labelActual + labelPredicted == 0)
			{
				accuracy  += 1;
				recall    += 1;
				precision += 1;
				fMeasure  += 1;
			}
			else
			{
				if (Utils.eq(forgivenessRate, 1.0)) accuracy += (setIntersection / setUnion);
				else accuracy += Math.pow(setIntersection / setUnion, forgivenessRate);

				if (labelPredicted > 0) precision += (setIntersection / labelPredicted);
				if (labelActual > 0)    recall += (setIntersection / labelActual);
				
			}
			hammingLoss += (symmetricDifference / numLabels);
		}

		// Set final values
		hammingLoss /= numInstances;
		accuracy /= numInstances;
		precision /= numInstances;
		recall /= numInstances;
		subsetAccuracy /= numInstances;
		fMeasure = computeF1Measure(precision, recall);
    }

  	private double computeF1Measure(double precision, double recall)
	{
	    if (Utils.eq(precision + recall, 0))
            return 0;
	    else
            return (2 * precision * recall) / (precision + recall);
	}
	
	public double getHammingLoss(){
		return hammingLoss;
	}
	
	public double getSubsetAccuracy(){
		return subsetAccuracy;
	}
	
	public double getAccuracy(){
		return accuracy;
	}
	
	public double getFMeasure(){
		return fMeasure;
	}
    
	public double getPrecision(){
		return precision;
	}
	
	public double getRecall(){
		return recall;
	}

}
