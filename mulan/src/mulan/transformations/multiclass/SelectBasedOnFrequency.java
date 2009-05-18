
package mulan.transformations.multiclass;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import mulan.core.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Class that implement the Select-Max transformation method.
 * @author Stavros
 */
public class SelectBasedOnFrequency extends MultiClassTransformationBase {

    SelectionType type;
    int[] labelOccurance;

    public SelectBasedOnFrequency(SelectionType type)
    {
        this.type = type;
    }

    @Override
    public Instances transformInstances(MultiLabelInstances mlData) throws Exception {
        // calculate label occurences
        Instances data = mlData.getDataSet();
        labelOccurance = new int[numOfLabels];
        int numInstances = data.numInstances();
        numPredictors = data.numAttributes() - numOfLabels;
        for (int i=0; i<numInstances; i++)
            for (int j=0; j<numOfLabels; j++)
            	if (data.instance(i).attribute(numPredictors + j).value((int) data.instance(i).value(numPredictors + j)).equals("1"))
            		labelOccurance[j]++;
        return super.transformInstances(mlData);
    }


    public List<Instance> transformInstance(Instance instance) {
        int value = labelOccurance[0];
        int labelSelected = 0;
        for (int labelIndex=1; labelIndex<numOfLabels; labelIndex++)
            if (instance.attribute(numPredictors + labelIndex).value((int) instance.value(numPredictors + labelIndex)).equals("1"))
            {
                boolean test = false;
                switch (type) {
                    case MIN : test = labelOccurance[labelIndex] < value ? true : false;
                               break;
                    case MAX : test = labelOccurance[labelIndex] > value ? true : false;
                               break;
                }

                if (test)
                {
                    value = labelOccurance[labelIndex];
                    labelSelected = labelIndex;
                }
            }

        double[] instanceValues = instance.toDoubleArray();
        double[] newValues = Arrays.copyOfRange(instanceValues, 0, numPredictors+1);
        newValues[numPredictors] = labelSelected;
        Instance tempInstance = new Instance(1, newValues);

        List<Instance> result = new ArrayList<Instance>();
        result.add(tempInstance);
        return result;
    }
}
