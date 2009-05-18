package mulan.attributeSelection;

import mulan.core.data.MultiLabelInstances;
import weka.attributeSelection.ASEvaluation;
import weka.core.Instances;

/**
 * 
 * @author greg
 */
public class Ranker extends weka.attributeSelection.Ranker
{
    private int numLabels;
    
    public int[] search(ASEvaluation ASEval, MultiLabelInstances mlData) throws Exception
    {
        numLabels = mlData.getNumLabels();
        return search(ASEval, mlData.getDataSet());
    }

    @Override
    public int[] search (ASEvaluation ASEval, Instances data) throws Exception 
    {
        String startSet = "" + data.numAttributes();
        for (int i=0; i<numLabels-1; i++)
            startSet += "," + (data.numAttributes()-i-1);
        setStartSet(startSet);
        return super.search(ASEval, data);
    }
}
