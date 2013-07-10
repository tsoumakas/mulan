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
 *    AverageRelativeRMSE.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.evaluation.measure;

import java.util.Arrays;

import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;

/**
 * Implementation of the Relative Root Mean Squared Error (RRMSE) measure.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2012.11.26
 */
public class AverageRelativeRMSE extends AverageRMSE implements MacroAverageMeasure {

    /** holds the average prediction MSE per target */
    private double[] averagePredictionMSE;
    /** holds the average value per target */
    private double[] targetAverages;

    public AverageRelativeRMSE(int numOfLabels, MultiLabelInstances train) {
        super(numOfLabels);
        averagePredictionMSE = new double[numOfLabels];
        targetAverages = new double[numOfLabels];
        int[] labelIndices = train.getLabelIndices();
        for (int i = 0; i < numOfLabels; i++) {
            targetAverages[i] = train.getDataSet().attributeStats(labelIndices[i]).numericStats.mean;
        }
    }

    public String getName() {
        return "Average Relative RMSE";
    }

    /**
     * Returns the value of the measure for each label
     * 
     * @param labelIndex the index of the label
     * @return the value of the measure
     */
    @Override
    public double getValue(int labelIndex) {
        double rmse = super.getValue(labelIndex);
        double rrmse = rmse / Math.sqrt(averagePredictionMSE[labelIndex] / counter[labelIndex]);
        return rrmse;
    }

    /**
     * When a target has missing values, they are ignored in RRMSE calculation.
     */
    @Override
    protected void updateInternal(MultiLabelOutput prediction, double[] truth) {
        double[] scores = prediction.getPvalues();
        for (int i = 0; i < truth.length; i++) {
            if (Double.isNaN(truth[i])) {
                continue;
            }
            counter[i]++;
            meanSquaredError[i] += (truth[i] - scores[i]) * (truth[i] - scores[i]);
            averagePredictionMSE[i] += (truth[i] - targetAverages[i])
                    * (truth[i] - targetAverages[i]);
        }
    }

    @Override
    public void reset() {
        super.reset();
        Arrays.fill(averagePredictionMSE, 0.0);
    }

}
