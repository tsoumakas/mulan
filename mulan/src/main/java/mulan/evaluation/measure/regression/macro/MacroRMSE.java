package mulan.evaluation.measure.regression.macro;

import mulan.classifier.MultiLabelOutput;
import mulan.evaluation.measure.MacroAverageMeasure;

/**
 * Implementation of the macro-averaged Root Mean Squared Error (RMSE) measure.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2014.11.07
 */
public class MacroRMSE extends MacroRegressionMeasureBase implements MacroAverageMeasure {

    public MacroRMSE(int numOfLabels) {
        super(numOfLabels);
    }

    public double getIdealValue() {
        return 0;
    }

    public String getName() {
        return "Macro RMSE";
    }

    public double getValue(int targetIndex) {
        double mse = error[targetIndex] / nonMissingCounter[targetIndex];
        double rmse = Math.sqrt(mse);
        return rmse;
    }

    public void updateInternal(MultiLabelOutput prediction, double[] truth) {
        double[] scores = prediction.getPvalues();
        for (int i = 0; i < truth.length; i++) {
            if (Double.isNaN(truth[i])) {
                continue;
            }
            nonMissingCounter[i]++;
            error[i] += (truth[i] - scores[i]) * (truth[i] - scores[i]);
        }
    }

}
