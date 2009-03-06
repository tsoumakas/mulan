
package mulan.transformations.multiclass;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import weka.core.Instance;

/**
 * Class that implement the Copy transformation method
 * @author Stavros
 */
public class Copy extends MultiClassTransformationBase {

    public Copy(int numOfLabels) {
        super(numOfLabels);
    }

    public List<Instance> transformInstance(Instance instance) {
        List<Instance> result = new ArrayList<Instance>();
        for (int labelIndex=0; labelIndex<numOfLabels; labelIndex++) {
            if (instance.attribute(numPredictors + labelIndex).value((int) instance.value(numPredictors + labelIndex)).equals("1")) {
                double[] instanceValues = instance.toDoubleArray();
                double[] newValues = Arrays.copyOfRange(instanceValues, 0, numPredictors+1);
                newValues[numPredictors] = labelIndex;
                Instance tempInstance = new Instance(1, newValues);
                result.add(tempInstance);
            }
        }
        return result;
    }
}
