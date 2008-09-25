/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package mulan.examples;


import java.io.FileReader;
import mulan.attributeSelection.LabelPowersetAttributeEvaluator;
import mulan.attributeSelection.Ranker;
import weka.attributeSelection.ChiSquaredAttributeEval;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 *
 * @author greg
 */
public class AttributeSelectionTest {
    public static void main(String[] args) throws Exception {
        final int NUM_LABELS = 6;
        String filename = "d:/work/datasets/multilabel/emotions/emotions.arff";
        FileReader frData = null;
        frData = new FileReader(filename);
        Instances data = new Instances(frData);                

        ChiSquaredAttributeEval csae = new ChiSquaredAttributeEval();
        LabelPowersetAttributeEvaluator lpae = new LabelPowersetAttributeEvaluator(NUM_LABELS);
        lpae.setAttributeEvaluator(csae);
        lpae.buildEvaluator(data);
        
        Ranker r = new Ranker(NUM_LABELS);
        int[] result = r.search(lpae, data);        
        //System.out.println(Arrays.toString(result));
        
        final int NUM_TO_KEEP=10;
        int[] toKeep = new int[NUM_TO_KEEP+NUM_LABELS];
        System.arraycopy(result, 0, toKeep, 0, NUM_TO_KEEP);
        for (int i=0; i<NUM_LABELS; i++)
            toKeep[NUM_TO_KEEP+i] = data.numAttributes()-1-i;
        
        Remove filterRemove = new Remove();
        filterRemove.setAttributeIndicesArray(toKeep);
        filterRemove.setInvertSelection(true);
        filterRemove.setInputFormat(data);
        Instances filtered = Filter.useFilter(data, filterRemove);
        
        //System.out.println(filtered.toString());
    }
}
