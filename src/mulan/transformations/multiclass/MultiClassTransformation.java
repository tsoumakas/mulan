
package mulan.transformations.multiclass;

import java.util.List;

import mulan.core.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;

/**
 * The interface for multi-class transformations methods.
 * 
 * @author Stavros
 */
public interface MultiClassTransformation {

    public Instances transformInstances(MultiLabelInstances dataSet) throws Exception;
     
    public List<Instance> transformInstance(Instance instance);
}
