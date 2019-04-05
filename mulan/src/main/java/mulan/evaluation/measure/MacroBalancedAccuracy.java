package mulan.evaluation.measure;
/**
 * Implementation of the macro-averaged balanced accuracy.
 * 
 * @author Bin Liu
 * @version 2018.12.12
 */

public class MacroBalancedAccuracy extends LabelBasedBipartitionMeasureBase implements MacroAverageMeasure{

	/**
     * Constructs a new object with given number of labels
     *
     * @param numOfLabels the number of labels
     */
    public MacroBalancedAccuracy(int numOfLabels) {
    	 super(numOfLabels);
    }


    @Override
    public String getName() {
        return "Macro-averaged Balanced Accuracy";
    }

    @Override
    public double getValue() {
        double sum = 0;
        int count = 0;
        for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
        	/*if(truePositives[labelIndex]+falseNegatives[labelIndex]==0)
        		continue;
        	*/
            sum += InformationRetrievalMeasures.balancedAccuracy
            	(truePositives[labelIndex],trueNegatives[labelIndex],
                 falsePositives[labelIndex], falseNegatives[labelIndex]);
            count++;
        }
        return sum / count;
    }

    /**
     * Returns the balanced accuracy for a label
     *
     * @param labelIndex the index of a label (starting from 0)
     * @return the balanced accuracy for the given label
     */
    @Override
    public double getValue(int labelIndex) {
        return InformationRetrievalMeasures.balancedAccuracy
            	(truePositives[labelIndex],trueNegatives[labelIndex],
                 falsePositives[labelIndex], falseNegatives[labelIndex]);
    }
    
    @Override
    public boolean handlesMissingValues(){
    	return true;
    }


	@Override
	public double getIdealValue() {
		return 1;
	}	
}

