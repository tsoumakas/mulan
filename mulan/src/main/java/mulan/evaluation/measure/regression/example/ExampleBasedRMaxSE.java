package mulan.evaluation.measure.regression.example;

import mulan.classifier.MultiLabelOutput;

/**
 * Computes the Root Max Squared Error (RMaxSE) in an example-based fashion.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2014.11.07
 */
public class ExampleBasedRMaxSE extends ExampleBasedRegressionMeasureBase {

    public String getName() {
        return "Example-based RMaxSE";
    }

    public double getIdealValue() {
        return 0;
    }

    public void updateInternal(MultiLabelOutput prediction, double[] truth) {
        int nonMissingCounter = 0;
        double maxAbsoluteError = 0;
        double[] scores = prediction.getPvalues();
        for (int i = 0; i < truth.length; i++) {
            if (Double.isNaN(truth[i])) {
                continue;
            }
            nonMissingCounter++;
            double error = Math.abs(truth[i] - scores[i]);
            if (error > maxAbsoluteError) {
                maxAbsoluteError = error;
            }
        }
        if (nonMissingCounter > 0) { // there is at least one non-missing target
            double rMaxSe = Math.sqrt(maxAbsoluteError * maxAbsoluteError);
            sum += rMaxSe;
            count++;
        }
    }
}
