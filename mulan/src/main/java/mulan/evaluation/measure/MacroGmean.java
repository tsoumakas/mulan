package mulan.evaluation.measure;
/**
 * Implementation of the macro-averaged G-mean.
 * 
 * @author Bin Liu
 * @version 2018.12.12
 */

public class MacroGmean extends LabelBasedBipartitionMeasureBase implements MacroAverageMeasure{

    /**
     * Constructs a new object with given number of labels
     *
     * @param numOfLabels the number of labels
     */
    public MacroGmean(int numOfLabels) {
    	 super(numOfLabels);
    }


    @Override
    public String getName() {
        return "Macro-averaged G-mean";
    }

    @Override
    public double getValue() {
        double sum = 0;
        int count = 0;
        for (int labelIndex = 0; labelIndex < numOfLabels; labelIndex++) {
        	/*if(truePositives[labelIndex]+falseNegatives[labelIndex]==0)
        		continue;
        	*/
        	
            sum += InformationRetrievalMeasures.gMean
            	(truePositives[labelIndex],trueNegatives[labelIndex],
                 falsePositives[labelIndex], falseNegatives[labelIndex]);
            count++;
        }
        return sum / count;
    }

    /**
     * Returns the G-mean for a label
     *
     * @param labelIndex the index of a label (starting from 0)
     * @return the G-mean for the given label
     */
    @Override
    public double getValue(int labelIndex) {
        return InformationRetrievalMeasures.gMean
            	(truePositives[labelIndex],trueNegatives[labelIndex],
                 falsePositives[labelIndex], falseNegatives[labelIndex]);
    }
    
    @Override
    public boolean handlesMissingValues(){
    	return true;
    }


	@Override
	public double getIdealValue() {
		// TODO Auto-generated method stub
		return 1;
	}	
}
