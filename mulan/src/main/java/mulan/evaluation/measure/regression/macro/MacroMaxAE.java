package mulan.evaluation.measure.regression.macro;

import mulan.classifier.MultiLabelOutput;
import mulan.evaluation.measure.MacroAverageMeasure;

/**
 * Implementation of the macro-averaged Root Maximum Squared Error (RMaxSE)
 * measure.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2014.11.07
 */
public class MacroMaxAE extends MacroRegressionMeasureBase implements MacroAverageMeasure {

    public MacroMaxAE(int numOfLabels) {
        super(numOfLabels);
    }

    public double getIdealValue() {
        return 0;
    }

    public String getName() {
        return "Macro MaxAE";
    }

    public double getValue(int targetIndex) {
        return error[targetIndex];
    }

    public void updateInternal(MultiLabelOutput prediction, double[] truth) {
        double[] scores = prediction.getPvalues();
        for (int i = 0; i < truth.length; i++) {
            if (Double.isNaN(truth[i])) {
                continue;
            }
            double absoluteError = Math.abs(truth[i] - scores[i]);
            if (absoluteError > error[i]) {
                error[i] = absoluteError;
            }
        }
    }

}
