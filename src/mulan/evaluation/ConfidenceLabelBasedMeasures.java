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
 *    ConfidenceLabelBasedMeasures.java
 *    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
 *
 */package mulan.evaluation;

import mulan.classifier.MultiLabelOutput;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.Utils;

public class ConfidenceLabelBasedMeasures {

    double[] auc = new double[2];
    double[] labelAUC;

    protected ConfidenceLabelBasedMeasures(MultiLabelOutput[] output, boolean[][] trueLabels) {
       computeMeasures(output, trueLabels);
    }

    ConfidenceLabelBasedMeasures(ConfidenceLabelBasedMeasures[] arrayOfMeasures) {
		int numLabels  = arrayOfMeasures[0].labelAUC.length;
		labelAUC  = new double[numLabels];

		for (ConfidenceLabelBasedMeasures measures : arrayOfMeasures)
		{
            for (Averaging type : Averaging.values()) {
                auc[type.ordinal()]  += measures.getAUC(type);
            }

			for(int labelIndex=0; labelIndex<numLabels; labelIndex++)
			{
				labelAUC[labelIndex]  += measures.getLabelAUC(labelIndex);
			}
		}

		int arrayLength = arrayOfMeasures.length;
        for (Averaging type : Averaging.values()) {
            auc[type.ordinal()]  /= arrayLength;
        }

		for(int labelIndex=0; labelIndex<numLabels; labelIndex++)
		{
			labelAUC[labelIndex]  /= arrayLength;
		}

    }
		
    private void computeMeasures(MultiLabelOutput[] output, boolean[][] trueLabels) {
        int numLabels = trueLabels[0].length;
        
        // AUC
        FastVector[] m_Predictions = new FastVector[numLabels];
		for (int j=0; j<numLabels; j++)
            m_Predictions[j] = new FastVector();
        FastVector all_Predictions = new FastVector();

        int numInstances = output.length;
        for (int instanceIndex=0; instanceIndex<numInstances; instanceIndex++)
		{
            double[] confidences = output[instanceIndex].getConfidences();
            for (int labelIndex = 0; labelIndex < numLabels; labelIndex++)
            {

                int classValue;
                boolean actual = trueLabels[instanceIndex][labelIndex];
                if (actual)
                    classValue = 1;
                else
                    classValue = 0;

                double[] dist = new double[2];
                dist[1] = confidences[labelIndex];
                dist[0] = 1 - dist[1];

                m_Predictions[labelIndex].addElement(new NominalPrediction(classValue, dist, 1));
                all_Predictions.addElement(new NominalPrediction(classValue, dist, 1));
            }
        }

        labelAUC = new double[numLabels];
        for (int i=0; i<numLabels; i++) {
            ThresholdCurve tc = new ThresholdCurve();
            Instances result = tc.getCurve(m_Predictions[i], 1);
            labelAUC[i] = ThresholdCurve.getROCArea(result);
		}
        auc[Averaging.MACRO.ordinal()] = Utils.mean(labelAUC);
        ThresholdCurve tc = new ThresholdCurve();
        Instances result = tc.getCurve(all_Predictions, 1);
        auc[Averaging.MICRO.ordinal()] = ThresholdCurve.getROCArea(result);
    }

    public double getLabelAUC(int label) {
        return labelAUC[label];
    }
	
	public double getAUC(Averaging averagingType){
		return auc[averagingType.ordinal()];
	}

}
