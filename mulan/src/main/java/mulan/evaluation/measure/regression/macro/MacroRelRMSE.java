package mulan.evaluation.measure.regression.macro;

import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;

/**
 * Implementation of the macro-averaged Relative Root Mean Squared Error
 * (RelRMSE) measure. RelRMSE for each target is equal to the RMSE of the
 * prediction divided by the RMSE of predicting the mean. Two versions of this
 * measure are implemented, one computes target means on the training set and
 * the other on the union of the training set and the test set. The first
 * version is the default one.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2014.11.07
 */
public class MacroRelRMSE extends MacroRelativeRegressionMeasureBase {

    public MacroRelRMSE(MultiLabelInstances train, MultiLabelInstances test) {
        super(train, test);
    }

    public String getName() {
        return "Macro RelRMSE";
    }

    public double getValue(int targetIndex) {
        double mse = error[targetIndex] / nonMissingCounter[targetIndex];
        double rel_mse = trainMeanPredError[targetIndex] / nonMissingCounter[targetIndex];
        double root_mse = Math.sqrt(mse);
        double root_rel_mse = Math.sqrt(rel_mse);
        double rrmse = root_mse / root_rel_mse;
        return rrmse;
    }

    public double getTotalSE(int labelIndex) {
        return error[labelIndex];
    }

    public double getTrainMeanTotalSE(int labelIndex) {
        return trainMeanPredError[labelIndex];
    }

    public double getFullMeanTotalSE(int labelIndex) {
        return fullMeanPredError[labelIndex];
    }

    public void updateInternal(MultiLabelOutput prediction, double[] truth) {
        double[] scores = prediction.getPvalues();
        for (int i = 0; i < truth.length; i++) {
            if (Double.isNaN(truth[i])) {
                continue;
            }
            nonMissingCounter[i]++;
            error[i] += (truth[i] - scores[i]) * (truth[i] - scores[i]);
            trainMeanPredError[i] += (truth[i] - targetMeansTrain[i])
                    * (truth[i] - targetMeansTrain[i]);
            fullMeanPredError[i] += (truth[i] - targetMeansFull[i])
                    * (truth[i] - targetMeansFull[i]);
        }
    }

    public double getIdealValue() {
        return 0;
    }

}
