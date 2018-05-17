package mulan.evaluation.measure.regression.macro;

import mulan.classifier.MultiLabelOutput;
import mulan.evaluation.measure.MacroAverageMeasure;

/**
 * Implementation of macro-averaged Mean Absolute Error (MAE) measure.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2014.11.07
 */
public class MacroMAE extends MacroRegressionMeasureBase implements MacroAverageMeasure {

    public MacroMAE(int numOfLabels) {
        super(numOfLabels);
    }

    public double getIdealValue() {
        return 0;
    }

    public String getName() {
        return "Macro MAE";
    }

    public double getValue(int targetIndex) {
        return error[targetIndex] / nonMissingCounter[targetIndex];
    }

    public void updateInternal(MultiLabelOutput prediction, double[] truth) {
        double[] scores = prediction.getPvalues();
        for (int i = 0; i < truth.length; i++) {
            if (Double.isNaN(truth[i])) {
                continue;
            }
            nonMissingCounter[i]++;
            error[i] += Math.abs(truth[i] - scores[i]);
        }
    }

}
