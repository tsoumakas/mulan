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
package mulan.evaluation.measure;

import java.util.Arrays;

import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;

/**
 * Implementation of the Relative Root Mean Squared Error (RRMSE) measure. RRMSE is equal to the
 * RMSE of the prediction divided by the RMSE of predicting the mean for its target. Two versions of
 * this measure are implemented, one computes target means on the training set and the other on the
 * union of the training set and the test set. The first version is the default one.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2013.07.26
 */
public class AverageRelativeRMSE extends AverageRMSE implements MacroAverageMeasure {

    /** holds the average (calculated of train set) prediction MSE per target */
    private double[] averageTrainPredictionMSE;
    /** holds the average (calculated of full dataset) prediction MSE per target */
    private double[] averageFullPredictionMSE;
    /** holds the average value per target in train set */
    private double[] targetAveragesTrain;
    /** holds the average value per target in full dataset */
    private double[] targetAveragesFull;

    public AverageRelativeRMSE(int numOfLabels, MultiLabelInstances train, MultiLabelInstances test) {
        super(numOfLabels);
        averageTrainPredictionMSE = new double[numOfLabels];
        averageFullPredictionMSE = new double[numOfLabels];
        targetAveragesTrain = new double[numOfLabels];
        targetAveragesFull = new double[numOfLabels];
        int[] labelIndices = train.getLabelIndices();
        for (int i = 0; i < numOfLabels; i++) {
            targetAveragesTrain[i] = train.getDataSet().attributeStats(labelIndices[i]).numericStats.mean;

            double testAverage = test.getDataSet().attributeStats(labelIndices[i]).numericStats.mean;
            int trainInstances = train.getDataSet().numInstances();
            int testInstances = test.getDataSet().numInstances();
            int allInstances = trainInstances + testInstances;
            targetAveragesFull[i] = (targetAveragesTrain[i] * trainInstances + testAverage
                    * testInstances)
                    / allInstances;
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

        double mse = meanSquaredError[labelIndex] / nonMissingCounter[labelIndex];
        double rel_mse = averageTrainPredictionMSE[labelIndex] / nonMissingCounter[labelIndex];

        double root_mse = Math.sqrt(mse);
        double root_rel_mse = Math.sqrt(rel_mse);

        double rrmse = root_mse / root_rel_mse;

        return rrmse;
    }

    public double getMSE(int labelIndex) {
        double mse = meanSquaredError[labelIndex];
        return mse;
    }

    public double getAvgMSETrain(int labelIndex) {
        double mse = averageTrainPredictionMSE[labelIndex];
        return mse;
    }

    public double getAvgMSEFull(int labelIndex) {
        double mse = averageFullPredictionMSE[labelIndex];
        return mse;
    }

    public double getTargetAverageFull(int labelIndex) {
        return targetAveragesFull[labelIndex];
    }

    public double getTargetAverageTrain(int labelIndex) {
        return targetAveragesTrain[labelIndex];
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
            nonMissingCounter[i]++;
            meanSquaredError[i] += (truth[i] - scores[i]) * (truth[i] - scores[i]);
            averageTrainPredictionMSE[i] += (truth[i] - targetAveragesTrain[i])
                    * (truth[i] - targetAveragesTrain[i]);
            averageFullPredictionMSE[i] += (truth[i] - targetAveragesFull[i])
                    * (truth[i] - targetAveragesFull[i]);
        }
    }

    @Override
    public void reset() {
        super.reset();
        Arrays.fill(averageTrainPredictionMSE, 0.0);
        Arrays.fill(averageFullPredictionMSE, 0.0);

    }

}
