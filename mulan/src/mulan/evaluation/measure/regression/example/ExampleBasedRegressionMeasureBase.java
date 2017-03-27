package mulan.evaluation.measure.regression.example;

import mulan.evaluation.measure.regression.RegressionMeasureBase;

/**
 * Base class for example-based regression measures
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2014.10.26
 *
 */
public abstract class ExampleBasedRegressionMeasureBase extends RegressionMeasureBase {

    /**
     * The current sum of the measure
     */
    protected double sum;
    /**
     * The number of validation examples processed
     */
    protected int count;

    public void reset() {
        sum = 0;
        count = 0;
    }

    public double getValue() {
        return sum / count;
    }

}
