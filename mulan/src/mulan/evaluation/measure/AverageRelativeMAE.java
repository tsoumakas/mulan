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
 * Implementation of the average Relative Mean Absolute Error (RMAE) measure. RMAE for each target
 * is equal to the MAE of the prediction divided by the MAE of predicting the mean. Two versions of
 * this measure are implemented, one computes target means on the training set and the other on the
 * union of the training set and the test set. The first version is the default one.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2013.11.08
 */
public class AverageRelativeMAE extends AverageMAE implements MacroAverageMeasure {

    /** holds the mean (calculated from train set) prediction's total absolute error per target */
    private double[] trainMeanPredTotalAbsoluteError;
    /** holds the mean (calculated from full set) prediction's total absolute error per target */
    private double[] fullMeanPredTotalAbsoluteError;
    /** holds the mean per target in train set */
    private double[] targetMeansTrain;
    /** holds the mean per target in full dataset */
    private double[] targetMeansFull;

    public AverageRelativeMAE(int numOfLabels, MultiLabelInstances train, MultiLabelInstances test) {
        super(numOfLabels);
        trainMeanPredTotalAbsoluteError = new double[numOfLabels];
        fullMeanPredTotalAbsoluteError = new double[numOfLabels];
        targetMeansTrain = new double[numOfLabels];
        targetMeansFull = new double[numOfLabels];
        int[] labelIndices = train.getLabelIndices();
        for (int i = 0; i < numOfLabels; i++) {
            targetMeansTrain[i] = train.getDataSet().attributeStats(labelIndices[i]).numericStats.mean;
            double testAverage = test.getDataSet().attributeStats(labelIndices[i]).numericStats.mean;
            int trainInstances = train.getDataSet().numInstances();
            int testInstances = test.getDataSet().numInstances();
            int allInstances = trainInstances + testInstances;
            targetMeansFull[i] = (targetMeansTrain[i] * trainInstances + testAverage
                    * testInstances)
                    / allInstances;
        }
    }

    public String getName() {
        return "Average Relative MAE";
    }

    /**
     * Returns the value of the measure for each target
     * 
     * @param targetIndex the index of the target
     * @return the value of the measure
     */
    @Override
    public double getValue(int targetIndex) {
        double mae = totalAbsoluteError[targetIndex] / nonMissingCounter[targetIndex];
        double rel_mae = trainMeanPredTotalAbsoluteError[targetIndex]
                / nonMissingCounter[targetIndex];
        double rmae = mae / rel_mae;
        return rmae;
    }

    public double getTotalAE(int labelIndex) {
        double mae = totalAbsoluteError[labelIndex];
        return mae;
    }

    public double getTrainMeanTotalAE(int labelIndex) {
        double mae = trainMeanPredTotalAbsoluteError[labelIndex];
        return mae;
    }

    public double getFullMeanTotalAE(int labelIndex) {
        double mae = fullMeanPredTotalAbsoluteError[labelIndex];
        return mae;
    }

    public double getTargetAverageFull(int labelIndex) {
        return targetMeansFull[labelIndex];
    }

    public double getTargetAverageTrain(int labelIndex) {
        return targetMeansTrain[labelIndex];
    }

    /**
     * When a target has missing values, they are ignored in RMAE calculation.
     */
    @Override
    protected void updateInternal(MultiLabelOutput prediction, double[] truth) {
        double[] scores = prediction.getPvalues();
        for (int i = 0; i < truth.length; i++) {
            if (Double.isNaN(truth[i])) {
                continue;
            }
            nonMissingCounter[i]++;
            totalAbsoluteError[i] += Math.abs(truth[i] - scores[i]);
            trainMeanPredTotalAbsoluteError[i] += Math.abs(truth[i] - targetMeansTrain[i]);
            fullMeanPredTotalAbsoluteError[i] += Math.abs(truth[i] - targetMeansFull[i]);
        }
    }

    @Override
    public void reset() {
        super.reset();
        Arrays.fill(trainMeanPredTotalAbsoluteError, 0.0);
        Arrays.fill(fullMeanPredTotalAbsoluteError, 0.0);

    }
}
