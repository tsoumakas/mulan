
package mulan.transformations.multiclass;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import weka.core.Instance;

/**
 * Class that implement the Ignore transformation method
 * @author Stavros
 */
public class Ignore extends MultiClassTransformationBase {

    public Ignore(int numOfLabels) {
        super(numOfLabels);
    }

    public List<Instance> transformInstance(Instance instance) {
        List<Instance> result = new ArrayList<Instance>();
        int indexOfSingleLabel = -1;
        int counter = 0;
        for (int labelIndex=0; labelIndex<numOfLabels; labelIndex++) {
            if (instance.attribute(numPredictors + labelIndex).value((int) instance.value(numPredictors + labelIndex)).equals("1")) {
                counter++;
                indexOfSingleLabel = labelIndex;
            }
            if (counter > 1 || counter == 0)
                return result;
        }

        double[] instanceValues = instance.toDoubleArray();
        double[] newValues = Arrays.copyOfRange(instanceValues, 0, numPredictors+1);
        newValues[numPredictors] = indexOfSingleLabel;
        Instance tempInstance = new Instance(1, newValues);
        result.add(tempInstance);
        return result;
    }
}
