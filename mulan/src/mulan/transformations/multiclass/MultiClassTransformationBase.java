
package mulan.transformations.multiclass;

import java.util.List;
import mulan.core.data.MultiLabelInstances;
import mulan.transformations.*;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

/**
 * The base class for multi-class transformation methods. It provides initial implementation
 * of {@link MultiClassTransformation} interface. All implementations of transformation 
 * methods should reuse this base class.
 *  
 * @author Stavros
 */
public abstract class MultiClassTransformationBase implements MultiClassTransformation {

    protected int numOfLabels;
    protected int numPredictors;
    protected int[] labelIndices;

    public Instances transformInstances(MultiLabelInstances mlData) throws Exception {
        labelIndices = mlData.getLabelIndices();
        numOfLabels = mlData.getNumLabels();
        Instances data = mlData.getDataSet();
        numPredictors = mlData.getDataSet().numAttributes()-numOfLabels;

        Instances transformed = new Instances(mlData.getDataSet(), 0);
        
        // delete all labels
        transformed = RemoveAllLabels.transformInstances(transformed, labelIndices);

        // add single label attribute
        FastVector classValues = new FastVector(numOfLabels);
        for(int x=0; x<numOfLabels; x++)
            classValues.addElement("Class"+(x+1));
        Attribute newClass = new Attribute("Class", classValues);
        transformed.insertAttributeAt(newClass, transformed.numAttributes());
        transformed.setClassIndex(transformed.numAttributes()-1);

        for (int instanceIndex=0; instanceIndex<data.numInstances(); instanceIndex++) {
            //System.out.println(data.instance(instanceIndex).toString());
            List<Instance> result = transformInstance(data.instance(instanceIndex));
            for (Instance instance : result)
            {
                //System.out.println(instance.toString());
                transformed.add(instance);
                //System.out.println(transformed.instance(transformed.numInstances()-1));
            }
        }
        return transformed;
    }

    
}
