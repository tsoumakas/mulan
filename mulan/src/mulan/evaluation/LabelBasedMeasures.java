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
 *    LabelBasedMeasures.java
 *    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
 *
 */

package mulan.evaluation;

import mulan.classifier.MultiLabelOutput;
import weka.core.Utils;

public class LabelBasedMeasures {

    // Macro and Micro averaged measures
    protected double[] recall    = new double[2];
    protected double[] precision = new double[2];
    protected double[] fMeasure  = new double[2];
    protected double[] accuracy  = new double[2];

    //Per label measures
	protected double[] labelRecall;
	protected double[] labelPrecision;
	protected double[] labelFMeasure;
	protected double[] labelAccuracy;


    protected LabelBasedMeasures(LabelBasedMeasures[] arrayOfMeasures) {

		int numLabels  = arrayOfMeasures[0].labelAccuracy.length;
		labelAccuracy  = new double[numLabels];
		labelRecall    = new double[numLabels];
		labelPrecision = new double[numLabels];
		labelFMeasure  = new double[numLabels];

		for(LabelBasedMeasures measures : arrayOfMeasures)
		{
            for (Averaging type : Averaging.values()) {
                accuracy[type.ordinal()]  += measures.getAccuracy(type);
                recall[type.ordinal()]    += measures.getRecall(type);
                precision[type.ordinal()] += measures.getPrecision(type);
                fMeasure[type.ordinal()]  += measures.getFMeasure(type);
            }

			for(int j=0; j<numLabels; j++)
			{
				labelAccuracy[j]  += measures.getLabelAccuracy(j);
				labelRecall[j]    += measures.getLabelRecall(j);
				labelPrecision[j] += measures.getLabelPrecision(j);
				labelFMeasure[j]  += measures.getLabelFMeasure(j);
			}
		}

		int arrayLength = arrayOfMeasures.length;
        for (Averaging type : Averaging.values()) {
            accuracy[type.ordinal()]  /= arrayLength;
            recall[type.ordinal()]    /= arrayLength;
            precision[type.ordinal()] /= arrayLength;
            fMeasure[type.ordinal()]  /= arrayLength;
        }

		for(int i=0; i<numLabels; i++)
		{
			labelAccuracy[i]  /= arrayLength;
			labelRecall[i]    /= arrayLength;
			labelPrecision[i] /= arrayLength;
			labelFMeasure[i]  /= arrayLength;
		}

    }

	protected LabelBasedMeasures(MultiLabelOutput[] output, boolean[][] trueLabels) {
        computeMeasures(output, trueLabels);
    }

    private void computeMeasures(MultiLabelOutput[] output, boolean[][] trueLabels) {
        int numLabels = trueLabels[0].length;

        //Counters are doubles to avoid typecasting
        //when performing divisions. It makes the code a
        //little cleaner but:
        //TODO: run performance tests on counting with doubles
        double[] falsePositives = new double[numLabels];
        double[] truePositives  = new double[numLabels];
        double[] falseNegatives = new double[numLabels];
        double[] trueNegatives  = new double[numLabels];

        //Count TP, TN, FP, FN
        int numInstances = output.length;
        for (int instanceIndex=0; instanceIndex<numInstances; instanceIndex++)
		{
            boolean[] bipartition = output[instanceIndex].getBipartition();

            for (int labelIndex = 0; labelIndex < numLabels; labelIndex++)
            {
                boolean actual = trueLabels[instanceIndex][labelIndex];
                boolean predicted = bipartition[labelIndex];

                if (actual && predicted)
                    truePositives[labelIndex]++;
                else if (!actual && !predicted)
                    trueNegatives[labelIndex]++;
                else if (predicted)
                    falsePositives[labelIndex]++;
                else
                    falseNegatives[labelIndex]++;
            }
        }

        // Evaluation measures for individual labels
        labelAccuracy  = new double[numLabels];
        labelRecall    = new double[numLabels];
        labelPrecision = new double[numLabels];
        labelFMeasure  = new double[numLabels];

        //Compute macro averaged measures
        for(int labelIndex = 0; labelIndex < numLabels; labelIndex++)
        {
            labelAccuracy[labelIndex] = (truePositives[labelIndex] + trueNegatives[labelIndex]) / numInstances;

            labelRecall[labelIndex] = truePositives[labelIndex] + falseNegatives[labelIndex] == 0 ? 0
                            :truePositives[labelIndex] / (truePositives[labelIndex] + falseNegatives[labelIndex]);

            labelPrecision[labelIndex] = truePositives[labelIndex] + falsePositives[labelIndex] == 0 ? 0
                            :truePositives[labelIndex] / (truePositives[labelIndex] + falsePositives[labelIndex]);

            labelFMeasure[labelIndex] = computeF1Measure(labelPrecision[labelIndex], labelRecall[labelIndex]);
        }
        accuracy[Averaging.MACRO.ordinal()]  = Utils.mean(labelAccuracy);
	    recall[Averaging.MACRO.ordinal()]    = Utils.mean(labelRecall);
	    precision[Averaging.MACRO.ordinal()] = Utils.mean(labelPrecision);
	    fMeasure[Averaging.MACRO.ordinal()]  = Utils.mean(labelFMeasure);

	    //Compute micro averaged measures
	    double tp = Utils.sum(truePositives);
	    double tn = Utils.sum(trueNegatives);
	    double fp = Utils.sum(falsePositives);
	    double fn = Utils.sum(falseNegatives);

	    accuracy[Averaging.MICRO.ordinal()]  = (tp + tn) / (numInstances * numLabels);
	    recall[Averaging.MICRO.ordinal()]    = tp + fn == 0 ? 0 : tp / (tp + fn);
	    precision[Averaging.MICRO.ordinal()] = tp + fp == 0 ? 0 : tp / (tp + fp);
	    fMeasure[Averaging.MICRO.ordinal()]  = computeF1Measure(precision[Averaging.MICRO.ordinal()], recall[Averaging.MICRO.ordinal()]);
    }

	
	public double getAccuracy(Averaging averagingType){
		return accuracy[averagingType.ordinal()];
	}
	
	public double getFMeasure(Averaging averagingType){
		return fMeasure[averagingType.ordinal()];
	}
	
	public double getRecall(Averaging averagingType){
		return recall[averagingType.ordinal()];
	}
	
	public double getPrecision(Averaging averagingType){
		return precision[averagingType.ordinal()];
	}


	public double getLabelAccuracy(int label){
		return labelAccuracy[label];
	}

	public double getLabelFMeasure(int label){
		return labelFMeasure[label];
	}

	public double getLabelRecall(int label){
		return labelRecall[label];
	}

	public double getLabelPrecision(int label){
		return labelPrecision[label];
	}

    private double computeF1Measure(double precision, double recall)
	{
	    if (Utils.eq(precision + recall, 0))
            return 0;
	    else
            return (2 * precision * recall) / (precision + recall);
	}
}
