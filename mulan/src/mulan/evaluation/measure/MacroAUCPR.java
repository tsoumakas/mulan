package mulan.evaluation.measure;

import weka.classifiers.evaluation.ThresholdCurve;
import weka.core.Instances;
import weka.core.Utils;

public class MacroAUCPR extends LabelBasedAUC implements MacroAverageMeasure{

	public MacroAUCPR(int numOfLabels) {
		super(numOfLabels);
		// TODO Auto-generated constructor stub
	}
	
    @Override
    public String getName() {
        return "Macro-averaged AUCPR";
    }

    @Override
    public double getValue() {
        double[] labelAUCPR = new double[numOfLabels];
        for (int i = 0; i < numOfLabels; i++) {
            ThresholdCurve tc = new ThresholdCurve();
            try{
                Instances result = tc.getCurve(m_Predictions[i], 1);
             // When the "m_Predictions[i].size()==0" is true, the return of "getCurve" function is "null" 
                labelAUCPR[i] = ThresholdCurve.getPRCArea(result);
            }
            catch (Exception e){  //when "result" is "null"
            	//e.printStackTrace();
            	labelAUCPR[i]=0.5;
            }
            
            //when the AUCPR is NaN (true labels only contain "1" or "0" values)
            if(Double.isNaN(labelAUCPR[i])){
            	labelAUCPR[i]=0.5;       
            }
        }
        return Utils.mean(labelAUCPR);
    }

    /**
     * Returns the AUCPR for a particular label
     * 
     * @param labelIndex the index of the label 
     * @return the AUCPR for that label
     */
    @Override
    public double getValue(int labelIndex) {
        ThresholdCurve tc = new ThresholdCurve();
        try{
        	Instances result = tc.getCurve(m_Predictions[labelIndex], 1);
        	return ThresholdCurve.getPRCArea(result);
        }
        catch (Exception e){
        	//e.printStackTrace();
        	return 0.5;
        }
        
          
    }
    
    @Override
    public boolean handlesMissingValues(){
    	return true;
    }

}
