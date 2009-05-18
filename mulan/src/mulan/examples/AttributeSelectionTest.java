/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package mulan.examples;


import java.util.Arrays;
import mulan.attributeSelection.LabelPowersetAttributeEvaluator;
import mulan.attributeSelection.Ranker;
import mulan.core.data.MultiLabelInstances;
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
        MultiLabelInstances data = new MultiLabelInstances(path + filestem + ".arff", path + filestem + ".xml");

        ChiSquaredAttributeEval csae = new ChiSquaredAttributeEval();
        LabelPowersetAttributeEvaluator lpae = new LabelPowersetAttributeEvaluator();
        lpae.setAttributeEvaluator(csae);
        lpae.buildEvaluator(data);
        
        Ranker r = new Ranker();
        int[] result = r.search(lpae, data);        
        System.out.println(Arrays.toString(result));
        
        final int NUM_TO_KEEP=10;
        int[] toKeep = new int[NUM_TO_KEEP+data.getNumLabels()];
        System.arraycopy(result, 0, toKeep, 0, NUM_TO_KEEP);
        for (int i=0; i<data.getNumLabels(); i++)
            toKeep[NUM_TO_KEEP+i] = data.getDataSet().numAttributes()-1-i;
        
        Remove filterRemove = new Remove();
        filterRemove.setAttributeIndicesArray(toKeep);
        filterRemove.setInvertSelection(true);
        filterRemove.setInputFormat(data.getDataSet());
        Instances filtered = Filter.useFilter(data.getDataSet(), filterRemove);
        
        System.out.println(filtered.toString());
    }
}
