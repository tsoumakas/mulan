package mulan.evaluation.measure.regression.macro;

import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;

/**
 * Implementation of the macro-averaged Relative Mean Absolute Error (RelMAE)
 * measure. RelMAE for each target is equal to the MAE of the prediction divided
 * by the MAE of predicting the mean. Two versions of this measure are
 * implemented, one computes target means on the training set and the other on
 * the union of the training set and the test set. The first version is the
 * default one.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2014.11.07
 */
public class MacroRelMAE extends MacroRelativeRegressionMeasureBase {

    public MacroRelMAE(MultiLabelInstances train, MultiLabelInstances test) {
        super(train, test);
    }

    public double getFullMeanTotalAE(int labelIndex) {
        return fullMeanPredError[labelIndex];
    }

    public double getIdealValue() {
        return 0;
    }

    public String getName() {
        return "Macro RelMAE";
    }

    public double getTotalAE(int labelIndex) {
        return error[labelIndex];
    }

    public double getTrainMeanTotalAE(int labelIndex) {
        return trainMeanPredError[labelIndex];
    }

    public double getValue(int targetIndex) {
        double mae = error[targetIndex] / nonMissingCounter[targetIndex];
        double rel_mae = trainMeanPredError[targetIndex] / nonMissingCounter[targetIndex];
        double rmae = mae / rel_mae;
        return rmae;
    }

    public void updateInternal(MultiLabelOutput prediction, double[] truth) {
        double[] scores = prediction.getPvalues();
        for (int i = 0; i < truth.length; i++) {
            if (Double.isNaN(truth[i])) {
                continue;
            }
            nonMissingCounter[i]++;
            error[i] += Math.abs(truth[i] - scores[i]);
            trainMeanPredError[i] += Math.abs(truth[i] - targetMeansTrain[i]);
            fullMeanPredError[i] += Math.abs(truth[i] - targetMeansFull[i]);
        }
    }
}
