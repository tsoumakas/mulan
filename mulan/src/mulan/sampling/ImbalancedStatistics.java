package mulan.sampling;

import java.io.Serializable;
import java.util.Arrays;

import mulan.data.LabelSet;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;

public class ImbalancedStatistics implements Serializable, TechnicalInformationHandler {

	public static void main(String[] args) {
		// TODO Auto-generated method stub

	}
	
	
	private static final long serialVersionUID = 1206845794397561644L; //
	
	private int c1[];  //number of instances associated with each label
	private int c0[];  //number of instances not associated with each label
    private double IRLbls[]=null;  //IRLbl of each label
    private double ImRs[]=null;  //ImR=c0/c1 of each label
    private double meanIRLb;  //mean value of IRLbls
    private double meanImR; //mean value of ImRs
  

	private double CVIR;
    private double maxIRLb;
    private double maxImR;
    private double minIRLb;
   	private double minImR;
    private double SCUMBLE;
    private double SCUMBLEs[];
    
    
    /**
	 * @return the c1
	 */
	public int[] getC1() {
		return c1;
	}


	/**
	 * @return the c0
	 */
	public int[] getC0() {
		return c0;
	}


	/**
	 * @return the iRLbls
	 */
	public double[] getIRLbls() {
		return IRLbls;
	}


	/**
	 * @return the imRs
	 */
	public double[] getImRs() {
		return ImRs;
	}

	public double getMeanImR() {
		return meanImR;
	}
	
	
	public double getMeanIRLb() {
		return meanIRLb;
	}


	/**
	 * @return the cVIR
	 */
	public double getCVIR() {
		return CVIR;
	}


	/**
	 * @return the maxIR
	 */
	public double getMaxIRLb() {
		return maxIRLb;
	}


	/**
	 * @return the maxImR
	 */
	public double getMaxImR() {
		return maxImR;
	}
	
	 /**
		 * @return the minIRLb
	 */
	public double getMinIRLb() {
		return minIRLb;
	}


	/**
	 * @return the minImR
	 */
	public double getMinImR() {
		return minImR;
	}	
	
	
	/**
	 * @return the sCUMBLE
	 */
	public double getSCUMBLE() {
		return SCUMBLE;
	}



	/**
	 * @return the sCUMBLEs
	 */
	public double[] getSCUMBLEs() {
		return SCUMBLEs;
	}


	public void calculateImSta(MultiLabelInstances mlDate){
    	int numInstances = mlDate.getNumInstances();
        int numLabels = mlDate.getNumLabels();
        int[] labelIndices = mlDate.getLabelIndices();
        Instances dataSet=mlDate.getDataSet();
        
        c1=new int[numLabels];
        c0=new int[numLabels];
        IRLbls = new double[numLabels];
    	ImRs = new double[numLabels];
        SCUMBLEs=new double[numInstances];
        
        
        meanIRLb=0;
        meanImR=0;
        CVIR=0;
        SCUMBLE=0;
        
        
        //calculate c1, c0
        for (int i = 0; i < numInstances; i++) {
        	Instance ins=dataSet.get(i);
        	for (int j = 0; j < numLabels; j++) {    
        		/* Returns the value of a nominal, string, date, 
        		  or relational attribute for the instance as a string. */
        		if (ins.stringValue(labelIndices[j]).equals("1")) {
                    c1[j]++;
                }
        		else{
        			c0[j]++;
        		}
            }
        }        
        
        //calculate IRLbls ImRs
        int mc=0;
        for (int j = 0; j < numLabels; j++) {
        	if(c1[j]>mc){
        		mc=c1[j];
        	}
        }
        
        for (int j = 0; j < numLabels; j++) {
        	if(c1[j]==0){	
        		IRLbls[j]=mc*1.0/0.1;
        		ImRs[j]=c0[j]*1.0/0.1;
        	}
        	else{
        		IRLbls[j]=mc*1.0/c1[j];
        		ImRs[j]=c0[j]*1.0/c1[j];
        	}
        }
        
        
        //calculate MeanIR, MaxIRLb, MaxImR, MinIRLb, MinImR
        maxIRLb=Double.MIN_VALUE;   maxImR=Double.MIN_VALUE;
        minIRLb=Double.MAX_VALUE;   minImR=Double.MAX_VALUE;
        int cIR=0,cImR=0;
        for (int j = 0; j < numLabels; j++) {
        	if(IRLbls[j]<Double.MAX_VALUE){
        		meanIRLb+=IRLbls[j];
        		if(maxIRLb<IRLbls[j]){
        			maxIRLb=IRLbls[j];
        		}
        		if(minIRLb>IRLbls[j]){
        			minIRLb=IRLbls[j];
        		}
        		cIR++;
        	}
        	
        	if(ImRs[j]<Double.MAX_VALUE){
        		meanImR+=ImRs[j];
        		if(maxImR<ImRs[j]){
        			maxImR=ImRs[j];
        		}
        		if(minImR>ImRs[j]){
        			minImR=ImRs[j];
        		}
        		cImR++;
        	}
        }
        meanIRLb/=cIR;
        meanImR/=cImR;
        
        //calculate CVIR
        for(double d:IRLbls){
        	if(d<Double.MAX_VALUE)
        		CVIR+=Math.pow((d-meanIRLb), 2.0);
    	}
        CVIR=Math.sqrt(CVIR/(cIR-1))/meanIRLb;
        
        //calculate SCUMBLE, SCUMBLEs
        SCUMBLE=0;
        for(int i=0;i<numInstances;i++){
        	Instance ins=dataSet.get(i);
        	double pro=1,ave=0;
        	int c=0;
        	for(int j=0;j<numLabels;j++){
        		if(ins.stringValue(labelIndices[j]).equals("1")){
        			pro*=IRLbls[j];  //Do not considering Double.MAX_VALUE
        			ave+=IRLbls[j];
        			c++;
        		}
        	}
        	if(c==0){	
        		SCUMBLEs[i]=0.0;
        	}
        	else{
            	ave/=c;
            	SCUMBLEs[i]=1-(1/ave)*Math.pow(pro, 1.0/c);
        	}

        	SCUMBLE+=SCUMBLEs[i];
        }
        SCUMBLE/=numInstances;
    }
    
	
	
	/* (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	@Override
	public String toString() {
		 StringBuilder sb = new StringBuilder();
		 sb.append("meanIRLb: ").append(meanIRLb).append("\n");
		 sb.append("CVIR: ").append(CVIR).append("\n");
		 sb.append("maxIRLb: ").append(maxIRLb).append("\n");
		 sb.append("minIRLb: ").append(minIRLb).append("\n");
		 sb.append("maxImR: ").append(maxImR).append("\n");
		 sb.append("minImR: ").append(minImR).append("\n");
		 sb.append("SCUMBLE: ").append(SCUMBLE).append("\n");
		 
		 sb.append("Measurements:\t").append("c1\t").append("c0\t").append("IRLbls\t").append("ImRs").append("\n");
	     for (int j = 0; j < c1.length; j++) {
	            sb.append("label ").append(j + 1).append(": \t").append(c1[j]+"\t").append(c0[j]+"\t").append(IRLbls[j]+"\t").append(ImRs[j]+"\t").append("\n");
	     }
	     
		 return sb.toString();
	}


	@Override
	public TechnicalInformation getTechnicalInformation() {
		// TODO Auto-generated method stub
		return null;
	}

}
