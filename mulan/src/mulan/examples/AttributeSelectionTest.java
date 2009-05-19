/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package mulan.examples;


import java.util.Arrays;
import mulan.attributeSelection.LabelPowersetAttributeEvaluator;
import mulan.attributeSelection.Ranker;
import mulan.core.data.MultiLabelInstances;
import weka.attributeSelection.AttributeEvaluator;
import weka.attributeSelection.ChiSquaredAttributeEval;
import weka.core.Instances;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author greg
 */
public class AttributeSelectionTest {
    
    public static void main(String[] args) throws Exception
    {
        String path = Utils.getOption("path", args);
        String filestem = Utils.getOption("filestem", args);
        MultiLabelInstances mlData = new MultiLabelInstances(path + filestem + ".arff", path + filestem + ".xml");

        AttributeEvaluator ae = new ChiSquaredAttributeEval();
        LabelPowersetAttributeEvaluator lpae = new LabelPowersetAttributeEvaluator(ae, mlData);
        
        Ranker r = new Ranker();
        int[] result = r.search(lpae, mlData);
        System.out.println(Arrays.toString(result));
        
        final int NUM_TO_KEEP=10;
        int[] toKeep = new int[NUM_TO_KEEP+mlData.getNumLabels()];
        System.arraycopy(result, 0, toKeep, 0, NUM_TO_KEEP);
        int[] labelIndices = mlData.getLabelIndices();
        for (int i=0; i<mlData.getNumLabels(); i++)
            toKeep[NUM_TO_KEEP+i] = labelIndices[i];
        
        Remove filterRemove = new Remove();
        filterRemove.setAttributeIndicesArray(toKeep);
        filterRemove.setInvertSelection(true);
        filterRemove.setInputFormat(mlData.getDataSet());
        Instances filtered = Filter.useFilter(mlData.getDataSet(), filterRemove);
        MultiLabelInstances mlFiltered = new MultiLabelInstances(filtered, mlData.getLabelsMetaData());

        // You can now work on the reduced multi-label dataset mlFiltered
    }
}
