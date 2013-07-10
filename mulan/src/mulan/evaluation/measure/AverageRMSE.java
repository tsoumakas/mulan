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
 *    AverageRMSE.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.evaluation.measure;

import java.util.Arrays;

import mulan.classifier.MultiLabelOutput;

/**
 * Implementation of the Root Mean Squared Error (RMSE) measure.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2012.11.26
 */
public class AverageRMSE extends RegressionMeasureBase implements MacroAverageMeasure {

    /** holds the MSE per target */
    protected double[] meanSquaredError;
    /** counts the number of non-missing values per target */
    protected int[] counter;

    public AverageRMSE(int numOfLabels) {
        meanSquaredError = new double[numOfLabels];
        counter = new int[numOfLabels];
    }

    public String getName() {
        return "Average RMSE";
    }

    public double getValue() {
        double value = 0;
        for (int i = 0; i < meanSquaredError.length; i++) {
            value += getValue(i);
        }
        return value / meanSquaredError.length;
    }

    /**
     * Returns the value of the measure for each label.
     * 
     * @param labelIndex the index of the label
     * @return the value of the measure
     */
    public double getValue(int labelIndex) {
        double rmse = Math.sqrt(meanSquaredError[labelIndex] / counter[labelIndex]);
        return rmse;
    }

    public double getIdealValue() {
        return 0;
    }

    /**
     * When a target has missing values, they are ignored in RMSE calculation.
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
        }
    }

    @Override
    public void reset() {
        Arrays.fill(counter, 0);
        Arrays.fill(meanSquaredError, 0.0);
    }

}
