
package mulan.transformations.multiclass;

import java.util.List;

import weka.core.Instance;
import weka.core.Instances;

/**
 * The interface for multi-class transformations methods.
 * 
 * @author Stavros
 */
public interface MultiClassTransformation {

    public Instances transformInstances(Instances dataSet) throws Exception;
     
    public List<Instance> transformInstance(Instance instance);
}
