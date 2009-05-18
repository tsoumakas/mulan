
package mulan.transformations.multiclass;

import java.util.List;
import weka.core.Instance;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * Class that implement the Select-Random transformation method
 * @author Stavros
 */

public class SelectRandom extends MultiClassTransformationBase {


    public List<Instance> transformInstance(Instance instance) {
        ArrayList<Integer> labels = new ArrayList<Integer>();
        for (int labelIndex=0; labelIndex<numOfLabels; labelIndex++)
            if (instance.attribute(numPredictors + labelIndex).value((int) instance.value(numPredictors + labelIndex)).equals("1"))
                labels.add(labelIndex);

        int randomLabel = labels.get((int)(Math.random()*labels.size()));
        double[] instanceValues = instance.toDoubleArray();
        double[] newValues = Arrays.copyOfRange(instanceValues, 0, numPredictors+1);
        newValues[numPredictors] = randomLabel;
        Instance tempInstance = new Instance(1, newValues);

        List<Instance> result = new ArrayList<Instance>();
        result.add(tempInstance);
        return result;
    }

}
