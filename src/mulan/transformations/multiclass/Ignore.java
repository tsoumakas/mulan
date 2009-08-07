
package mulan.transformations.multiclass;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import mulan.transformations.RemoveAllLabels;
import weka.core.Instance;

/**
 * Class that implement the Ignore transformation method
 * @author Stavros
 */
public class Ignore extends MultiClassTransformationBase {

    /**
     * Transforms a multi-label example with a single annotation to a
     * single-label example and ignores multi-label example with more
     * annotations
     *
     * @param instance a multi-label example
     * @return a list that is either empty or contains the transformed
     * single-label example
     */
    public List<Instance> transformInstance(Instance instance) {
        List<Instance> result = new ArrayList<Instance>();
        int indexOfSingleLabel = -1;
        int counter = 0;
        for (int labelCounter=0; labelCounter<numOfLabels; labelCounter++) {
            int index = labelIndices[labelCounter];
            if (instance.attribute(index).value((int) instance.value(index)).equals("1"))
            {
                counter++;
                indexOfSingleLabel = labelCounter;
            }
            if (counter > 1)
                break;
        }
        if (counter > 1 || counter == 0)
            return result;

        Instance transformedInstance;
        try {
            transformedInstance = RemoveAllLabels.transformInstance(instance, labelIndices);
            transformedInstance.setDataset(null);
            transformedInstance.insertAttributeAt(transformedInstance.numAttributes());
            transformedInstance.setValue(transformedInstance.numAttributes()-1, indexOfSingleLabel);
            result.add(transformedInstance);
        } catch (Exception ex) {
            Logger.getLogger(Ignore.class.getName()).log(Level.SEVERE, null, ex);
        }
        return result;
    }

}
