package mulan.evaluation.measure.regression.macro;

import java.util.Arrays;

import mulan.evaluation.measure.MacroAverageMeasure;
import mulan.evaluation.measure.regression.RegressionMeasureBase;

public abstract class MacroRegressionMeasureBase extends RegressionMeasureBase implements
        MacroAverageMeasure {

    /**
     * holds the (total or max) squared or absolute or logistic error per target
     */
    protected double[] error;
    /** counts the number of non-missing values per target */
    protected int[] nonMissingCounter;

    public MacroRegressionMeasureBase(int numOfLabels) {
        error = new double[numOfLabels];
        nonMissingCounter = new int[numOfLabels];
    }

    /** returns the number of non missing values for this target **/
    public int getNonMissing(int labelIndex) {
        return nonMissingCounter[labelIndex];
    }

    public double getValue() {
        double value = 0;
        for (int i = 0; i < error.length; i++) {
            value += getValue(i);
        }
        return value / error.length;
    }

    public void reset() {
        Arrays.fill(error, 0.0);
        Arrays.fill(nonMissingCounter, 0);
    }
}
