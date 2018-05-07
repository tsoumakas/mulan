package mulan.evaluation.measure.regression;

import java.io.Serializable;

import mulan.classifier.MultiLabelOutput;
import mulan.core.ArgumentNullException;
import mulan.evaluation.GroundTruth;
import mulan.evaluation.measure.Measure;
import weka.core.SerializedObject;

/**
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2014.11.07
 */
public abstract class RegressionMeasureBase implements Measure, Serializable {

    public final void update(MultiLabelOutput prediction, GroundTruth truth) {
        if (prediction == null) {
            throw new ArgumentNullException("Prediction is null");
        }
        if (truth == null) {
            throw new ArgumentNullException("Ground truth is null");
        }
        if (!truth.isRegression()) {
            throw new ArgumentNullException("Classification ground truth is null");
        }

        updateInternal(prediction, truth.getTrueValues());
    }

    /**
     * Returns a string with the value of a measure
     * 
     * @return string representation of the value of a measure
     */
    @Override
    public String toString() {
        double value = Double.NaN;
        try {
            value = getValue();
        } catch (Exception ex) {
        }
        return getName() + ": " + String.format("%.4f", value);
    }

    /**
     * Updates the measure based on an example
     * 
     * @param prediction the output of the algorithm for the example
     * @param truth the ground truth of the example
     */
    public abstract void updateInternal(MultiLabelOutput prediction, double[] truth);

    public Measure makeCopy() throws Exception {
        return (Measure) new SerializedObject(this).getObject();
    }

    /**
     * By default, regression measures should handle missing ground truth values. This method should be
     * overridden if a particular measure's implementation cannot handle missing ground truth values.
     */
    public boolean handlesMissingValues() {
        return true;
    }
}
