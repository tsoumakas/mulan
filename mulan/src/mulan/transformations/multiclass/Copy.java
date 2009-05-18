
package mulan.transformations.multiclass;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import mulan.transformations.RemoveAllLabels;
import weka.core.Instance;

/**
 * Class that implement the Copy transformation method
 * @author Stavros
 * @author Grigorios Tsoumakas
 */
public class Copy extends MultiClassTransformationBase {


    public List<Instance> transformInstance(Instance instance) {
        List<Instance> result = new ArrayList<Instance>();
        for (int counter=0; counter<numOfLabels; counter++) {
            if (instance.attribute(labelIndices[counter]).value((int) instance.value(labelIndices[counter])).equals("1"))
            {
                Instance transformed = null;
                try {
                    transformed = RemoveAllLabels.transformInstance(instance, labelIndices);
                    transformed.setDataset(null);
                    transformed.insertAttributeAt(transformed.numAttributes());
                    transformed.setValue(transformed.numAttributes()-1, counter);
                } catch (Exception ex) {
                    Logger.getLogger(Copy.class.getName()).log(Level.SEVERE, null, ex);
                }
                result.add(transformed);
            }
        }
        return result;
    }

}
